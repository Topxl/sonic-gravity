/**
 * app.js — la boucle fermée.
 *
 *   son réel → spectres pré-fader → champ → force → moteurs → faders → son réel
 *
 * L'application ne sait pas si la surface est un écran ou une X-Touch : elle ne
 * parle que le vocabulaire MCU (cf `surface.js`).
 */

import { Scene } from "./field.js";
import { MixWorld } from "./mix.js";
import { Haptics, MotorFader, normalizeForce } from "./motor.js";
import { VirtualSurface } from "./surface.js";

const PROFILE_HZ = 12; // le relief du champ n'a pas besoin des 60 Hz de l'écran
const PROFILE_STEPS = 36;

/**
 * Met la force du champ à une échelle utilisable par un moteur.
 *
 * Le gradient a une échelle arbitraire, qui dépend du morceau et de l'état du
 * mix. Une constante en dur donnerait une console molle sur un mix et brutale
 * sur le suivant. On calibre donc sur la scène au démarrage, puis on suit
 * lentement — avec un plancher, sans lequel un mix DÉJÀ bon (forces minuscules)
 * verrait ses miettes amplifiées jusqu'à sembler une vraie gravité.
 */
class ForceScaler {
  constructor() {
    this.ref = 0;
    this.floor = 0;
  }
  update(forces) {
    let m = 0;
    for (let i = 0; i < forces.length; i++) m = Math.max(m, Math.abs(forces[i]));
    if (!m) return this.ref || 1;
    if (!this.ref) {
      this.ref = m;
      this.floor = m * 0.35;
    } else {
      this.ref += (m - this.ref) * 0.02;
      this.ref = Math.max(this.ref, this.floor);
    }
    return this.ref;
  }
}

class App {
  constructor() {
    this.canvas = document.getElementById("surface");
    this.surface = new VirtualSurface(this.canvas, (e) => this.onSurfaceEvent(e));
    this.haptics = new Haptics();
    this.scaler = new ForceScaler();
    this.motors = [];
    this.assist = 0.7;
    this.running = false;
    this.gravityOn = true;
    this.lastFrame = 0;
    this.lastProfile = 0;
    this.scene = null;
    this.parts = null;
  }

  async boot() {
    this.spec = await (await fetch("spec.json")).json();
    this.mix = new MixWorld(this.spec);
    const manifest = await this.mix.load();

    this.n = this.mix.size;
    this.gains = new Float64Array(this.n).fill(0.75);
    // Ancre = ce que la MAIN a posé. La gravité corrige autour de cette
    // intention ; sans elle, le minimum du potentiel est un mix entièrement
    // recalculé et les faders partent en butée.
    this.anchor = new Float64Array(this.n).fill(0.75);
    this.audible = new Float64Array(this.n).fill(1);
    this.levelCalibrated = false;
    for (let i = 0; i < this.n; i++) {
      this.motors.push(new MotorFader(0.75));
      this.surface.setLabel(i, this.mix.tracks[i].label);
      this.surface.setFader(i, 0.75);
    }
    this.surface.setLabel(8, "MASTER");
    this.surface.setFader(8, 0.9);

    const fitted = this.spec.target.fitted;
    document.getElementById("source").textContent = manifest.source;
    document.getElementById("target-info").textContent = fitted
      ? `gabarit mesuré sur ${this.spec.target.nTracks} morceaux`
      : "gabarit par défaut (non mesuré)";

    this.bindUi();
    this.surface.render();
    requestAnimationFrame((t) => this.frame(t));
  }

  bindUi() {
    document.getElementById("play").addEventListener("click", () => this.toggle());
    document.getElementById("gravity").addEventListener("change", (e) => {
      this.gravityOn = e.target.checked;
      if (!this.gravityOn) this.haptics.stop();
    });
    const assist = document.getElementById("assist");
    assist.addEventListener("input", (e) => {
      this.assist = Number(e.target.value) / 100;
      document.getElementById("assist-val").textContent = `${e.target.value} %`;
    });
    // Liberté = ce qu'on laisse la machine reprendre. À 0 elle corrige autour
    // du geste ; à 100 elle recalcule le mix entier et les faders vont chercher
    // leur optimum, butées comprises. C'est le curseur qui dit ce qu'est
    // l'outil, il mérite d'être sous la main plutôt que dans un fichier.
    const freedom = document.getElementById("freedom");
    this.anchorBase = this.spec.weights.anchor ?? 1;
    freedom.addEventListener("input", (e) => {
      const f = Number(e.target.value) / 100;
      this.spec.weights.anchor = this.anchorBase * (1 - f);
      document.getElementById("freedom-val").textContent =
        f === 0 ? "tenue" : f >= 0.99 ? "libre" : `${e.target.value} %`;
    });
    document.getElementById("scramble").addEventListener("click", () => this.scramble());
    document.getElementById("release").addEventListener("click", () => this.releaseAll());
  }

  async toggle() {
    const btn = document.getElementById("play");
    if (this.running) {
      this.mix.stop();
      this.running = false;
      btn.textContent = "Écouter";
      this.haptics.stop();
      return;
    }
    await this.mix.start();
    this.running = true;
    this._playSince = 0;
    btn.textContent = "Arrêter";
  }

  /** Casse volontairement le mix : c'est de là que la gravité doit le sortir. */
  scramble() {
    for (let i = 0; i < this.n; i++) {
      const v = 0.15 + Math.random() * 0.85;
      this.gains[i] = v;
      this.motors[i].value = v;
      this.motors[i].velocity = 0;
    }
  }

  /** Tout lâcher : les faders tombent dans la vallée du champ. */
  releaseAll() {
    for (const m of this.motors) {
      m.held = false;
      m.velocity = 0;
    }
  }

  onSurfaceEvent(e) {
    const t = this.mix.tracks[e.ch];
    switch (e.type) {
      case "touch": {
        const m = this.motors[e.ch];
        if (!m) return;
        if (e.pressed) {
          m.grab(e.value);
        } else {
          m.release(e.velocity ?? 0);
          // La main a fini son geste : c'est LÀ que l'intention se fige. La
          // reposer pendant le mouvement ferait fuir l'ancre avec le doigt et
          // le rappel ne s'opposerait plus à rien.
          this.anchor[e.ch] = m.fingerValue;
        }
        break;
      }
      case "fader": {
        const m = this.motors[e.ch];
        if (!m) return;
        m.move(e.value);
        break;
      }
      case "mute":
        if (t && e.pressed) {
          t.muted = !t.muted;
          this.surface.setMuteLed(e.ch, t.muted);
        }
        break;
      case "solo":
        if (t && e.pressed) {
          t.soloed = !t.soloed;
          this.surface.setSoloLed(e.ch, t.soloed);
        }
        break;
      case "select":
        if (e.pressed) {
          for (let i = 0; i < this.n; i++) this.surface.setSelectLed(i, i === e.ch);
        }
        break;
      case "vpot":
        // Pas encore câblé à un paramètre : l'anneau bouge, le son ne change
        // pas. Un V-Pot qui agirait en silence sur le son serait pire.
        if (t) {
          const s = this.surface.state[e.ch];
          this.surface.setVpotRing(e.ch, Math.min(1, Math.max(0, s.vpot + e.delta * 0.05)));
        }
        break;
      default:
        break;
    }
  }

  frame(now) {
    requestAnimationFrame((t) => this.frame(t));
    const dt = this.lastFrame ? Math.min(0.1, (now - this.lastFrame) / 1000) : 0;
    this.lastFrame = now;

    if (this.running) {
      const fresh = this.mix.sample(now);
      this.mix.audibility(this.audible);
      // Pendant un silence (bouclage, break), on garde la dernière scène : un
      // spectre vide donnerait un potentiel absurde et secouerait les faders.
      //
      // ⚠️ Sans scène du tout, on n'en fabrique pas une : les spectres valent
      // encore leur valeur d'initialisation, et le potentiel calculé dessus
      // sortait à 88 sur la toute première image — un pic unique, invisible à
      // l'œil, mais qui salissait toute mesure faite sur la session.
      if (!fresh) {
        this.surface.render();
        return;
      }
      // La scène recalcule les formes spectrales et les recouvrements par
      // paire : ~1 300 opérations pour six tranches, négligeable, et c'est ce
      // qui rend les centaines d'évaluations du relief quasi gratuites.
      this.scene = new Scene(this.spec, this.mix.P);
    }

    if (this.scene) {
      this.calibrateLevel(now);
      const raw = this.scene.force(this.gains, this.audible, this.anchor);
      const ref = this.scaler.update(raw);
      this.parts = this.scene.potential(this.gains, this.audible, this.anchor);

      for (let i = 0; i < this.n; i++) {
        const m = this.motors[i];
        const f = this.gravityOn ? normalizeForce(raw[i], ref) : 0;
        this.gains[i] = m.step(f, this.assist, dt);
        this.surface.setFader(i, m.value);
        this.surface.setForce(i, m.force, m.fingerValue, m.held);
        this.surface.setMeter(i, Math.min(1, this.mix.tracks[i].level * 2.2));
        if (m.held) this.haptics.feel(m.force, m.takeCrossing(), now);
      }
      this.mix.applyGains(this.gains, this.audible);

      if (now - this.lastProfile > 1000 / PROFILE_HZ) {
        this.lastProfile = now;
        this.updateProfiles();
      }
      this.updateReadout();
    }

    this.surface.render();
  }

  /**
   * Cale la cible de niveau sur la scène réellement chargée.
   *
   * Une cible en dur ne veut rien dire : elle dépend de l'échelle des stems,
   * de la loi de fader et du morceau. Mesurée à −20 dB en dur alors que la
   * scène en sortait −25,1, elle poussait les six faders dans le MÊME sens
   * pour rejoindre un chiffre arbitraire. On prend donc le niveau atteint au
   * point neutre, une fois les spectres stabilisés.
   */
  calibrateLevel(now) {
    if (this.levelCalibrated || !this.running) return;
    if (!this._playSince) this._playSince = now;
    if (now - this._playSince < 2000) return;
    const at = this.scene.potential(this.anchor, this.audible, null);
    this.spec.target.levelDb = at.levelDb;
    this.levelCalibrated = true;
    document.getElementById("v-level").title = `cible calée à ${at.levelDb.toFixed(1)} dB`;
  }

  /** Relief du potentiel le long de chaque course, normalisé pour l'affichage. */
  updateProfiles() {
    for (let i = 0; i < this.n; i++) {
      if (!this.audible[i]) {
        this.surface.setFieldProfile(i, null);
        continue;
      }
      const p = this.scene.profile(i, this.gains, this.audible, this.anchor, PROFILE_STEPS);
      // Normalisation sur un QUANTILE, pas sur le maximum. Couper un fader
      // coûte énormément, donc le maximum est toujours la butée basse : la
      // ramener à 1 écrasait toute la variation utile, et une vallée large et
      // plate s'affichait comme un aplat vert sans fond visible.
      const sorted = Float64Array.from(p).sort();
      const lo = sorted[0];
      const hi = sorted[Math.floor(sorted.length * 0.7)];
      const span = Math.max(hi - lo, 1e-9);
      const norm = new Float64Array(p.length);
      for (let k = 0; k < p.length; k++) norm[k] = Math.min(1, (p[k] - lo) / span);
      this.surface.setFieldProfile(i, norm);
    }
  }

  updateReadout() {
    const p = this.parts;
    if (!p) return;
    document.getElementById("v-total").textContent = p.total.toFixed(3);
    document.getElementById("v-tilt").textContent = p.tilt.toFixed(3);
    document.getElementById("v-mask").textContent = p.mask.toFixed(3);
    document.getElementById("v-level").textContent = `${p.levelDb.toFixed(1)} dB`;
    const bar = document.getElementById("v-bar");
    // 0 = mix dans le domaine des vrais morceaux, 1 = très loin.
    bar.style.width = `${Math.min(100, (p.total / 6) * 100)}%`;
  }
}

// Exposée pour les vérifications automatisées (Playwright) et pour inspecter
// le champ depuis la console du navigateur. Aucune logique n'en dépend.
const app = new App();
window.__gravity = app;

app.boot().catch((e) => {
  document.getElementById("error").textContent = String(e);
  document.getElementById("error").hidden = false;
  console.error(e);
});
