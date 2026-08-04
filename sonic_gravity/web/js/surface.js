/**
 * surface.js — une Behringer X-Touch à l'écran.
 *
 * ⚠️ Contrat : cette classe expose la MÊME API que `web/src/lib/mcu.ts` du
 * projet Yantra (`setFader`, `setMuteLed`, `setVpotRing`, `setMeter`, …) et
 * émet les mêmes événements (`{type:"fader"|"mute"|"solo"|…}`). Le jour où la
 * vraie console est branchée, on substitue l'implémentation et RIEN d'autre ne
 * bouge : l'application ne sait pas si elle parle à un écran ou à du matériel.
 * Toute méthode ajoutée ici qui n'existe pas côté MCU casserait cette
 * promesse — sauf celles marquées « écran seulement », qui sont du rendu pur
 * (le relief du champ, l'écart doigt/capuchon) et n'ont pas d'équivalent
 * matériel parce que le matériel, lui, se ressent.
 *
 * Tout est peint dans UN canvas : neuf faders, huit vu-mètres et le relief du
 * champ redessinés à 60 Hz coûteraient bien plus cher en DOM, et le hit-testing
 * manuel donne le multi-touch sans se battre contre les événements natifs.
 */

const STRIPS = 8; // une X-Touch a 8 tranches + le master
const MASTER = 8;

const C = {
  bg: "#0a0c10",
  panel: "#12161c",
  panelEdge: "#1e242d",
  ink: "#e6e9ef",
  ink2: "#8b94a3",
  ink3: "#525b69",
  accent: "#a78bfa",
  good: "#34d399",
  bad: "#f87171",
  rail: "#0d1014",
  cap: "#d8dde6",
  capHeld: "#ffffff",
  meter: "#4ade80",
  meterHot: "#fbbf24",
  led: "#ef4444",
  ledSolo: "#fbbf24",
  ledSel: "#60a5fa",
};

const BTN = [
  { key: "rec", label: "REC", color: C.led },
  { key: "solo", label: "SOLO", color: C.ledSolo },
  { key: "mute", label: "MUTE", color: C.accent },
  { key: "select", label: "SEL", color: C.ledSel },
];

// Sur téléphone, une tranche fait 66 px : quatre boutons en grille 2×2 donnent
// des cibles de 31×26, très en dessous des 44 px de la charte tactile. On ne
// garde donc que les deux qui servent vraiment ici, empilés sur toute la
// largeur — REC et SEL n'ont aucun effet dans cette démo, et un bouton qui ne
// fait rien ne mérite pas de voler la place d'un bouton qui agit.
const BTN_COMPACT = BTN.filter((b) => b.key === "mute" || b.key === "solo");

// Extension de la zone d'ENTRÉE, sans un pixel d'encombrement en plus (patron
// `.tap-44-y` du projet Yantra). En hauteur seulement : étendre en largeur
// volerait les clics de la tranche voisine.
const TAP_MIN = 44;

export class VirtualSurface {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {(e:object)=>void} onEvent  reçoit les mêmes événements que `Mcu`
   */
  constructor(canvas, onEvent) {
    this.canvas = canvas;
    this.onEvent = onEvent;
    this.ctx = canvas.getContext("2d");
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);

    /** Décalage de banque — le mécanisme MCU d'origine, qui sert ici à tenir
     * sur un téléphone : 4 tranches visibles au lieu de 8, sans jamais
     * introduire de défilement (règle UX : les contrôles ne scrollent pas). */
    this.bankOffset = 0;
    this.visibleStrips = STRIPS;

    this.state = [];
    for (let i = 0; i <= MASTER; i++) {
      this.state.push({
        fader: 0.75,
        finger: 0.75,
        held: false,
        force: 0,
        meter: 0,
        vpot: 0.5,
        label: "",
        sub: "",
        mute: false,
        solo: false,
        select: false,
        rec: false,
        profile: null, // écran seulement : relief du potentiel le long de la course
        active: false,
      });
    }
    this.transport = { play: false, rec: false };
    this.assistLabel = "";

    this._rects = { faders: [], buttons: [], vpots: [], banks: [] };
    this._pointers = new Map();
    this._bind();
    this.resize();
  }

  // ── Géométrie ─────────────────────────────────────────────────────────────
  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(320, rect.width);
    const h = Math.max(360, rect.height);
    this.canvas.width = Math.round(w * this.dpr);
    this.canvas.height = Math.round(h * this.dpr);
    this.w = w;
    this.h = h;
    // Sous 720 px on ne peut pas donner 44 px de cible à huit tranches ET au
    // master : on montre une demi-console et on navigue par banque.
    this.visibleStrips = w < 720 ? 4 : STRIPS;
    if (this.bankOffset + this.visibleStrips > STRIPS) {
      this.bankOffset = Math.max(0, STRIPS - this.visibleStrips);
    }
    this._layout();
  }

  _layout() {
    const pad = 10;
    const bankH = this.visibleStrips < STRIPS ? 34 : 0;
    const usableW = this.w - pad * 2;
    const n = this.visibleStrips + 1; // + master
    const gap = 6;
    const stripW = (usableW - gap * (n - 1)) / n;
    const top = pad + bankH;
    const stripH = this.h - top - pad;

    this.geo = { pad, gap, stripW, stripH, top, bankH, n };
    this._rects = { faders: [], buttons: [], vpots: [], banks: [] };

    if (bankH) {
      const bw = 92;
      this._rects.banks.push({ dir: -1, x: pad, y: pad, w: bw, h: bankH - 6 });
      this._rects.banks.push({ dir: 1, x: pad + bw + 8, y: pad, w: bw, h: bankH - 6 });
    }

    for (let s = 0; s < n; s++) {
      const isMaster = s === n - 1;
      const ch = isMaster ? MASTER : this.bankOffset + s;
      const x = pad + s * (stripW + gap);
      const y = top;

      // Répartition verticale d'une tranche, de haut en bas.
      const scribbleH = 26;
      const vpotR = Math.min(stripW * 0.3, 19);
      const vpotH = isMaster ? 0 : vpotR * 2 + 10;
      const compact = this.visibleStrips < STRIPS;
      const btns = compact ? BTN_COMPACT : BTN;
      const perRow = compact ? 1 : 2;
      const rows = btns.length / perRow;
      const btnH = isMaster ? 0 : compact ? 32 : 26;
      const btnGap = 4;
      const btnBlockH = isMaster ? 0 : btnH * rows + btnGap * (rows - 1);
      // Course bornée : sur un grand écran, laisser le fader occuper toute la
      // hauteur restante donnait 570 px de course — une console étirée qui ne
      // ressemble plus au geste réel (100 mm sur une vraie X-Touch).
      const blockTop = y + scribbleH + vpotH + btnBlockH + 10;
      const avail = y + stripH - blockTop - 12;
      const faderH = Math.min(430, Math.max(90, avail));
      // Centré dans ce qui reste : borner la course sans recentrer laissait un
      // vide de 140 px sous la console sur un grand écran.
      const faderY = blockTop + (avail - faderH) / 2;

      if (!isMaster) {
        this._rects.vpots.push({
          ch,
          cx: x + stripW / 2,
          cy: y + scribbleH + 6 + vpotR,
          r: vpotR,
        });
        const bw = perRow === 1 ? stripW : (stripW - btnGap) / 2;
        btns.forEach((b, i) => {
          const bx = x + (i % perRow) * (bw + btnGap);
          const by = y + scribbleH + vpotH + Math.floor(i / perRow) * (btnH + btnGap);
          // La zone d'entrée déborde en hauteur jusqu'à 44 px, bornée à la
          // moitié de la gouttière pour ne jamais mordre sur le bouton voisin.
          const grow = Math.min((TAP_MIN - btnH) / 2, btnGap / 2);
          this._rects.buttons.push({
            ch,
            key: b.key,
            label: b.label,
            color: b.color,
            x: bx,
            y: by,
            w: bw,
            h: btnH,
            hitY: by - Math.max(0, grow),
            hitH: btnH + 2 * Math.max(0, grow),
          });
        });
      }

      this._rects.faders.push({
        ch,
        isMaster,
        x,
        y: faderY,
        w: stripW,
        h: faderH,
        stripX: x,
        stripY: y,
        stripH,
        scribbleH,
      });
    }
  }

  // ── Entrée : pointeur → événements MCU ────────────────────────────────────
  _bind() {
    const c = this.canvas;
    c.style.touchAction = "none";
    c.addEventListener("pointerdown", (e) => this._down(e));
    c.addEventListener("pointermove", (e) => this._move(e));
    c.addEventListener("pointerup", (e) => this._up(e));
    c.addEventListener("pointercancel", (e) => this._up(e));
    window.addEventListener("resize", () => this.resize());
  }

  _pos(e) {
    const r = this.canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  _valueFromY(f, y) {
    const inner = f.h - 18;
    const t = 1 - (y - (f.y + 9)) / inner;
    return Math.min(1, Math.max(0, t));
  }

  _down(e) {
    const { x, y } = this._pos(e);

    for (const b of this._rects.banks) {
      if (x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) {
        this.bankOffset = Math.max(
          0,
          Math.min(STRIPS - this.visibleStrips, this.bankOffset + b.dir * this.visibleStrips)
        );
        this._layout();
        this.onEvent({ type: "nav", dir: b.dir, unit: "bank", pressed: true });
        return;
      }
    }

    for (const b of this._rects.buttons) {
      const hy = b.hitY ?? b.y;
      const hh = b.hitH ?? b.h;
      if (x >= b.x && x <= b.x + b.w && y >= hy && y <= hy + hh) {
        this.canvas.setPointerCapture?.(e.pointerId);
        this._pointers.set(e.pointerId, { kind: "button", ref: b });
        this.onEvent({ type: b.key, ch: b.ch, pressed: true });
        return;
      }
    }

    for (const v of this._rects.vpots) {
      const dx = x - v.cx;
      const dy = y - v.cy;
      if (dx * dx + dy * dy <= (v.r + 8) * (v.r + 8)) {
        this.canvas.setPointerCapture?.(e.pointerId);
        this._pointers.set(e.pointerId, { kind: "vpot", ref: v, lastY: y });
        return;
      }
    }

    for (const f of this._rects.faders) {
      if (x >= f.x && x <= f.x + f.w && y >= f.y && y <= f.y + f.h) {
        this.canvas.setPointerCapture?.(e.pointerId);
        const value = this._valueFromY(f, y);
        this._pointers.set(e.pointerId, { kind: "fader", ref: f, lastY: y, lastT: e.timeStamp });
        // `touch` dit à l'application que la main tient le fader : c'est ce que
        // fait la détection capacitive d'une vraie X-Touch, et c'est ce qui
        // bascule le moteur de « pousser le fader » à « pousser contre le doigt ».
        this.onEvent({ type: "touch", ch: f.ch, pressed: true, value });
        this.onEvent({ type: "fader", ch: f.ch, value });
        return;
      }
    }
  }

  _move(e) {
    const p = this._pointers.get(e.pointerId);
    if (!p) return;
    const { x, y } = this._pos(e);
    if (p.kind === "fader") {
      const value = this._valueFromY(p.ref, y);
      const dt = Math.max(1, e.timeStamp - p.lastT);
      p.velocity = ((p.lastY - y) / (p.ref.h - 18) / dt) * 1000;
      p.lastY = y;
      p.lastT = e.timeStamp;
      this.onEvent({ type: "fader", ch: p.ref.ch, value });
    } else if (p.kind === "vpot") {
      // Encodeur relatif, comme le vrai V-Pot : ce qui compte est le
      // déplacement, jamais la position absolue du doigt.
      const delta = (p.lastY - y) / 6;
      if (Math.abs(delta) >= 1) {
        p.lastY = y;
        this.onEvent({ type: "vpot", ch: p.ref.ch, delta: Math.trunc(delta) });
      }
    }
  }

  _up(e) {
    const p = this._pointers.get(e.pointerId);
    if (!p) return;
    this._pointers.delete(e.pointerId);
    this.canvas.releasePointerCapture?.(e.pointerId);
    if (p.kind === "button") {
      this.onEvent({ type: p.ref.key, ch: p.ref.ch, pressed: false });
    } else if (p.kind === "fader") {
      this.onEvent({ type: "touch", ch: p.ref.ch, pressed: false, velocity: p.velocity || 0 });
    }
  }

  // ── Sortie : API identique à `Mcu` ────────────────────────────────────────
  setFader(ch, value) {
    this.state[ch].fader = Math.min(1, Math.max(0, value));
  }
  setMeter(ch, level) {
    this.state[ch].meter = Math.min(1, Math.max(0, level));
  }
  setVpotRing(ch, value) {
    this.state[ch].vpot = Math.min(1, Math.max(0, value));
  }
  setMuteLed(ch, on) {
    this.state[ch].mute = !!on;
  }
  setSoloLed(ch, on) {
    this.state[ch].solo = !!on;
  }
  setSelectLed(ch, on) {
    this.state[ch].select = !!on;
  }
  setRecLed(ch, on) {
    this.state[ch].rec = !!on;
  }
  setTransportLed(name, on) {
    this.transport[name] = !!on;
  }
  reset() {
    for (let ch = 0; ch <= MASTER; ch++) {
      this.setFader(ch, 0);
      this.setMuteLed(ch, false);
      this.setSoloLed(ch, false);
      this.setSelectLed(ch, false);
      this.setRecLed(ch, false);
    }
  }

  // ── Écran seulement (pas d'équivalent matériel : là-bas, ça se ressent) ───
  setLabel(ch, label, sub = "") {
    this.state[ch].label = label;
    this.state[ch].sub = sub;
    this.state[ch].active = !!label;
  }
  /** Relief du potentiel le long de la course, normalisé 0 (vallée) .. 1 (colline). */
  setFieldProfile(ch, profile) {
    this.state[ch].profile = profile;
  }
  /** Force appliquée [−1, 1] et position réelle du doigt, pour montrer l'écart. */
  setForce(ch, force, finger, held) {
    const s = this.state[ch];
    s.force = force;
    s.finger = finger;
    s.held = held;
  }

  // ── Rendu ─────────────────────────────────────────────────────────────────
  render() {
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, this.w, this.h);

    for (const b of this._rects.banks) this._drawBank(b);
    for (const f of this._rects.faders) this._drawStrip(f);
    for (const b of this._rects.buttons) this._drawButton(b);
    for (const v of this._rects.vpots) this._drawVpot(v);
  }

  _roundRect(x, y, w, h, r) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  _drawBank(b) {
    const ctx = this.ctx;
    const can = b.dir < 0 ? this.bankOffset > 0 : this.bankOffset + this.visibleStrips < STRIPS;
    ctx.fillStyle = C.panel;
    this._roundRect(b.x, b.y, b.w, b.h, 6);
    ctx.fill();
    ctx.strokeStyle = C.panelEdge;
    ctx.stroke();
    ctx.fillStyle = can ? C.ink : C.ink3;
    ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(b.dir < 0 ? "◀ BANK" : "BANK ▶", b.x + b.w / 2, b.y + b.h / 2);
  }

  _drawStrip(f) {
    const ctx = this.ctx;
    const s = this.state[f.ch];

    ctx.fillStyle = C.panel;
    this._roundRect(f.stripX, f.stripY, f.w, f.stripH, 8);
    ctx.fill();
    ctx.strokeStyle = C.panelEdge;
    ctx.lineWidth = 1;
    ctx.stroke();

    // Scribble strip : l'écran de la tranche.
    const sc = { x: f.stripX + 4, y: f.stripY + 4, w: f.w - 8, h: f.scribbleH - 8 };
    ctx.fillStyle = s.active ? "#1a2030" : "#0f1218";
    this._roundRect(sc.x, sc.y, sc.w, sc.h, 4);
    ctx.fill();
    ctx.fillStyle = s.active ? C.ink : C.ink3;
    ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(
      (s.label || (f.isMaster ? "MASTER" : "—")).toUpperCase().slice(0, 9),
      sc.x + sc.w / 2,
      sc.y + sc.h / 2
    );

    this._drawFader(f, s);
  }

  _drawFader(f, s) {
    const ctx = this.ctx;
    const railW = 5;
    const meterW = 8;
    const inner = f.h - 18;
    const top = f.y + 9;

    // Le relief occupe TOUTE la largeur utile de la tranche, le vu-mètre est
    // réservé à droite. Première version : le relief tenait dans 20 px puis le
    // rail était peint par-dessus, n'en laissant que deux liserés de 6 px — la
    // seule chose qui distingue cette console d'une console ordinaire était
    // masquée par son propre rail.
    const fieldX = f.x + 6;
    const fieldW = f.w - 12 - (meterW + 5);
    const cx = fieldX + fieldW / 2;
    const mx = f.x + f.w - 6 - meterW;

    if (s.profile && s.profile.length > 1) {
      const n = s.profile.length;
      for (let k = 0; k < n; k++) {
        // profile[0] = fader en bas ; l'écran a son origine en haut.
        const t = Math.min(1, Math.max(0, s.profile[n - 1 - k]));
        const y0 = top + (k / n) * inner;
        const y1 = top + ((k + 1) / n) * inner;
        const r = Math.round(52 + 196 * t);
        const g = Math.round(211 - 98 * t);
        const b = Math.round(153 - 40 * t);
        // La colline doit se voir autant que la vallée : à alpha 0,1 elle était
        // invisible et tout paraissait uniformément vert.
        ctx.fillStyle = `rgba(${r},${g},${b},${0.13 + 0.3 * Math.abs(1 - 2 * t)})`;
        ctx.fillRect(fieldX, y0, fieldW, y1 - y0 + 0.6);
      }

      // Et par-dessus, la COURBE du potentiel : un dégradé dit « plutôt bon
      // ici », une courbe dit exactement où est le fond et à quel point il est
      // creux. C'est elle qui rend le champ lisible d'un coup d'œil.
      ctx.beginPath();
      for (let k = 0; k < n; k++) {
        const t = Math.min(1, Math.max(0, s.profile[n - 1 - k]));
        const y = top + (k / (n - 1)) * inner;
        const x = fieldX + 3 + t * (fieldW - 6);
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = "rgba(255,255,255,0.42)";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Le fond de la vallée : là où le fader ira si on le lâche.
      let best = 0;
      for (let k = 1; k < n; k++) if (s.profile[k] < s.profile[best]) best = k;
      const by = top + (1 - best / (n - 1)) * inner;
      ctx.strokeStyle = C.good;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(fieldX, by);
      ctx.lineTo(fieldX + fieldW, by);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Rail, fin, par-dessus le relief.
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    this._roundRect(cx - railW / 2, top, railW, inner, 3);
    ctx.fill();

    // Vu-mètre, dans son couloir réservé.
    ctx.fillStyle = "#0d1014";
    this._roundRect(mx, top, meterW, inner, 3);
    ctx.fill();
    const mh = inner * Math.min(1, s.meter);
    if (mh > 1) {
      ctx.fillStyle = s.meter > 0.86 ? C.meterHot : C.meter;
      this._roundRect(mx, top + inner - mh, meterW, mh, 3);
      ctx.fill();
    }

    const yOf = (v) => top + (1 - v) * inner;

    // Écart doigt ↔ capuchon : la résistance, rendue visible.
    if (s.held && Math.abs(s.force) > 0.02) {
      const yf = yOf(s.finger);
      const yc = yOf(s.fader);
      ctx.strokeStyle = s.force > 0 ? C.good : C.bad;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx, yf);
      ctx.lineTo(cx, yc);
      ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.beginPath();
      ctx.arc(cx, yf, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Capuchon : dans le couloir du champ, pas sur toute la tranche — sinon il
    // recouvre le vu-mètre.
    const capH = 18;
    const capW = fieldW;
    const y = yOf(s.fader) - capH / 2;
    const x = fieldX;
    ctx.fillStyle = s.held ? C.capHeld : C.cap;
    this._roundRect(x, y, capW, capH, 4);
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x + 3, y + capH / 2);
    ctx.lineTo(x + capW - 3, y + capH / 2);
    ctx.stroke();

    // Liseré de force sur le capuchon : la direction où le moteur pousse.
    if (Math.abs(s.force) > 0.02) {
      ctx.fillStyle = s.force > 0 ? C.good : C.bad;
      const hb = Math.min(capH / 2 - 2, Math.abs(s.force) * (capH / 2));
      if (s.force > 0) ctx.fillRect(x, y, capW, hb);
      else ctx.fillRect(x, y + capH - hb, capW, hb);
    }
  }

  _drawButton(b) {
    const ctx = this.ctx;
    const s = this.state[b.ch];
    const on = s[b.key];
    ctx.fillStyle = on ? b.color : "#161b22";
    this._roundRect(b.x, b.y, b.w, b.h, 5);
    ctx.fill();
    ctx.strokeStyle = on ? b.color : C.panelEdge;
    ctx.stroke();
    ctx.fillStyle = on ? "#0a0c10" : C.ink2;
    ctx.font = "600 9px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(b.label, b.x + b.w / 2, b.y + b.h / 2);
  }

  _drawVpot(v) {
    const ctx = this.ctx;
    const s = this.state[v.ch];
    ctx.strokeStyle = C.panelEdge;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(v.cx, v.cy, v.r, 0, Math.PI * 2);
    ctx.stroke();

    // Anneau de LED : 11 points, comme le vrai V-Pot.
    const N = 11;
    const a0 = Math.PI * 0.75;
    const span = Math.PI * 1.5;
    const lit = Math.round(s.vpot * (N - 1));
    for (let k = 0; k < N; k++) {
      const a = a0 + (k / (N - 1)) * span;
      const px = v.cx + Math.cos(a) * (v.r - 3);
      const py = v.cy + Math.sin(a) * (v.r - 3);
      ctx.fillStyle = k === lit ? C.accent : "#242b35";
      ctx.beginPath();
      ctx.arc(px, py, k === lit ? 2.4 : 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = "#1a1f27";
    ctx.beginPath();
    ctx.arc(v.cx, v.cy, v.r - 7, 0, Math.PI * 2);
    ctx.fill();
  }
}
