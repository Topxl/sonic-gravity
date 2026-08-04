/**
 * motor.js — physique d'un fader motorisé, et ce qu'on en ressent.
 *
 * Sur une X-Touch, un fader est un moteur à courant continu doublé d'une
 * détection capacitive : la console sait si un doigt le tient. Ces deux états
 * ne se pilotent pas de la même façon, et c'est toute la différence entre un
 * gadget et une interface haptique :
 *
 *   • LÂCHÉ — le fader est un point matériel dans le champ. Il accélère selon
 *     la force, freine par frottement, et se pose au fond de la vallée. C'est
 *     le moment où la console bouge toute seule.
 *   • TENU  — le moteur ne peut pas déplacer le doigt, il pousse CONTRE lui.
 *     Le capuchon prend du retard sur la main : cet écart EST la résistance.
 *     Sur une vraie console on la sent dans les doigts ; à l'écran on la voit,
 *     et sur un téléphone la vibration la rend perceptible.
 *
 * Le pas d'intégration est fixe et découplé de la boucle d'affichage : une
 * image sautée ne doit pas changer la physique, sinon le fader part plus vite
 * sur une machine chargée.
 */

const DT = 1 / 120; // pas d'intégration, indépendant du taux de rafraîchissement
const MAX_SUBSTEPS = 8; // au-delà, l'onglet était en arrière-plan : on ne rattrape pas

export const DEFAULTS = {
  // Masse et frottement : réglés pour une course pleine en ~400 ms sous force
  // maximale, ce qui est l'ordre de grandeur d'un vrai fader motorisé.
  mass: 0.055,
  damping: 5.2,
  maxSpeed: 3.0,
  // Écart maximal entre le doigt et le capuchon, en fraction de course.
  compliance: 0.085,
  // Sous ce niveau de force, le moteur ne fait rien : sans seuil, les faders
  // frémissent en permanence et la console a l'air cassée.
  deadZone: 0.04,
};

/** Compresse une force d'échelle inconnue vers [−1, 1] en gardant son signe. */
export function normalizeForce(f, reference) {
  const r = Math.max(reference, 1e-9);
  return Math.tanh(f / r);
}

export class MotorFader {
  constructor(value = 0.75, opts = {}) {
    this.value = value;
    this.velocity = 0;
    this.held = false;
    this.fingerValue = value;
    this.force = 0; // force normalisée [−1, 1], telle qu'appliquée
    this.opts = { ...DEFAULTS, ...opts };
    this._acc = 0;
    this._lastSign = 0;
    this._crossed = false; // a-t-on traversé l'équilibre depuis la dernière lecture
  }

  /** Le doigt se pose : on mémorise l'écart pour que le capuchon ne saute pas. */
  grab(value) {
    this.held = true;
    this.fingerValue = value;
    this.velocity = 0;
  }

  /** Le doigt bouge. */
  move(value) {
    this.fingerValue = Math.min(1, Math.max(0, value));
  }

  /** Le doigt se lève : le fader repart avec la vitesse du geste, pas à zéro. */
  release(flickVelocity = 0) {
    this.held = false;
    this.velocity = Math.max(-1, Math.min(1, flickVelocity));
  }

  /**
   * Avance la physique.
   * @param {number} force  force normalisée [−1, 1] (positif = vers le haut)
   * @param {number} assist poids donné à la gravité, 0 = console inerte
   * @param {number} dt     temps écoulé depuis l'appel précédent, en secondes
   */
  step(force, assist, dt) {
    const o = this.opts;
    let f = force * assist;
    if (Math.abs(f) < o.deadZone) f = 0;
    this.force = f;

    // Traversée de l'équilibre : c'est le « cran » qu'on veut faire sentir.
    const sign = Math.sign(f);
    if (sign !== 0 && this._lastSign !== 0 && sign !== this._lastSign) this._crossed = true;
    if (sign !== 0) this._lastSign = sign;

    if (this.held) {
      // Le moteur pousse contre la main : le capuchon prend du retard sur le
      // doigt, proportionnellement à la force. Aucune intégration ici — la
      // main impose la position, la gravité ne fait que la déporter.
      const target = this.fingerValue + f * o.compliance;
      this.value = Math.min(1, Math.max(0, target));
      this.velocity = 0;
      return this.value;
    }

    this._acc += dt;
    let steps = 0;
    while (this._acc >= DT && steps < MAX_SUBSTEPS) {
      const a = (f - o.damping * this.velocity * o.mass) / o.mass;
      this.velocity += a * DT;
      this.velocity = Math.max(-o.maxSpeed, Math.min(o.maxSpeed, this.velocity));
      this.value += this.velocity * DT;
      if (this.value <= 0) {
        this.value = 0;
        this.velocity = 0;
      } else if (this.value >= 1) {
        this.value = 1;
        this.velocity = 0;
      }
      this._acc -= DT;
      steps++;
    }
    if (steps >= MAX_SUBSTEPS) this._acc = 0; // onglet revenu du fond : on repart net
    return this.value;
  }

  /** Vrai une seule fois par traversée du point d'équilibre. */
  takeCrossing() {
    const c = this._crossed;
    this._crossed = false;
    return c;
  }
}

/**
 * Retour haptique réel sur les appareils qui en ont un.
 *
 * `navigator.vibrate` ne sait pas moduler l'intensité : on module donc la
 * CADENCE et la durée des impulsions, ce qui se ressent comme une texture —
 * une résistance rugueuse quand on pousse vers un mauvais mix, une impulsion
 * sèche au passage du point d'équilibre.
 */
export class Haptics {
  constructor() {
    this.enabled = typeof navigator !== "undefined" && typeof navigator.vibrate === "function";
    this._next = 0;
  }

  /** @param {number} force force normalisée appliquée, [−1, 1] */
  feel(force, crossed, now) {
    if (!this.enabled) return;
    if (crossed) {
      navigator.vibrate(12); // le cran : sec, court, non répété
      this._next = now + 120;
      return;
    }
    const mag = Math.abs(force);
    if (mag < 0.25 || now < this._next) return;
    // Plus la force est grande, plus les impulsions se rapprochent : de ~200 ms
    // (frottement léger) à ~55 ms (butée franche).
    const period = 200 - 145 * Math.min(1, (mag - 0.25) / 0.75);
    navigator.vibrate(Math.round(4 + 8 * mag));
    this._next = now + period;
  }

  stop() {
    if (this.enabled) navigator.vibrate(0);
  }
}
