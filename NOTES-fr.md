# Gravité sonore

Une console de mixage dont les faders **résistent quand on va vers un mauvais
mix**, et **glissent tout seuls vers le bon** quand on les lâche.

Pas un générateur de musique : un **exosquelette pour la main du mixeur**. La
machine ne décide pas du mix, elle rend sensible la forme du problème.

Le contrôleur visé est une **Behringer X-Touch** (9 faders motorisés, protocole
MCU). En attendant le matériel, la surface est reproduite à l'écran avec
**exactement la même API** — le jour où la console est branchée, on substitue
l'implémentation et rien d'autre ne bouge.

## Lancer

```bash
python -m sonic_gravity.prepare_demo --list          # morceaux décomposés disponibles
python -m sonic_gravity.prepare_demo e9cb4ebd80a11f31  # extrait une boucle de 24 s
python -m sonic_gravity.serve                          # → http://127.0.0.1:8700
```

Puis : **Écouter**, **Dérégler**, **Tout lâcher**. Les faders retombent dans la
vallée. Prendre un fader à la souris : il traîne derrière le doigt, du côté où
la gravité pousse.

Recaler le gabarit sur d'autres morceaux :

```bash
python -m sonic_gravity.fit_target --limit 220
```

## Comment ça marche

```
son réel → spectres pré-fader → champ → force → moteurs → faders → son réel
```

Chaque tranche est mesurée **pré-fader** : `P[i,b]` = puissance de la tranche
`i` dans la bande `b`, indépendante de son gain. Le mix vaut alors

```
M[b] = Σᵢ (gᵢ·aᵢ)² · P[i,b]
```

ce qui rend le potentiel **dérivable analytiquement** par rapport aux faders.
La force envoyée à chaque moteur est `−∂V/∂gᵢ`.

C'est là qu'est l'intérêt : la gravité répond à « que se passerait-il si je
poussais ce fader ? » **sans le pousser**. Impossible à mesurer en direct sur
neuf faders — il faudrait bouger chacun et écouter. C'est exactement ce qu'un
modèle du monde conditionné par l'action sait faire.

### Le potentiel V

| Terme | Ce qu'il mesure | Pourquoi |
|---|---|---|
| `tilt` | Écart au gabarit spectral des vrais morceaux, en z-score par bande, passé en **perte de Huber** | Le gabarit est **mesuré** sur 220 morceaux de la bibliothèque, pas postulé. Huber empêche un break de dominer le champ |
| `mask` | Masquage entre tranches (intersection des formes spectrales × énergies) | **Invariant d'échelle** : le gradient dit « baisse celui qui se bat », jamais « baisse tout » |
| `level` | Écart au niveau visé, **poids faible** | Le niveau est le rôle du master, pas des tranches. À poids fort, le minimum partait en butée haute |
| `anchor` | Rappel vers la position posée par la main | Ce qui sépare l'exosquelette du pilote automatique. Curseur « Liberté » à fond = la machine reprend tout le mix |

Chaque bande est pondérée par la **matière réellement disponible** : pendant un
break où seul le piano joue, aucun fader ne peut ramener du grave, donc les
bandes graves cessent de compter. Cette pondération se mesure à plein gain, elle
est donc constante vis-à-vis de la dérivation.

### Ce que le gradient garantit, et ce qu'il ne garantit pas

L'approximation centrale est **`M[b] = Σ g²·P[i,b]`, c'est-à-dire des phases
décorrélées**. Elle tient bien sur des stems de séparateur ; elle tomberait sur
deux copies du même signal. La compression, la saturation et le masquage
psychoacoustique ne sont pas modélisés non plus.

Cette référence analytique n'est donc pas la vérité : c'est **la ligne de base à
battre** par un monde appris, qui est la suite du projet (voir plus bas).

## Vérifications

```bash
python -m pytest tests/test_sonic_gravity_field.py tests/test_sonic_gravity_parity.py
```

- **43 tests sur le champ**, dont le gradient comparé à des différences finies
  centrées sur 26 configurations tirées au sort (avec mute, avec ancrage, à la
  jonction de Huber). C'est la seule preuve qui compte : une erreur de signe ou
  de facteur ne casse aucun test de forme, elle pousse simplement la main vers
  le mauvais mix, en silence. Une telle erreur a été trouvée par ce test —
  `DB_SCALE` (facteur de la dérivée) écrit à la place de `10` (facteur de la
  valeur), qui gonflait la force de 1,7× à 2,8× selon la scène.
- **6 tests de parité Python ↔ JavaScript** : potentiel, chaque terme, gradient
  et découpage en bandes, à 10⁻⁹ près. Le modèle est entraîné d'un côté et
  appliqué de l'autre — une divergence ne planterait pas, elle se découvrirait
  des semaines plus tard, à l'oreille.

Mesuré en direct dans le navigateur : **60 fps**, force + potentiel en **1 µs**,
les six reliefs (216 évaluations du potentiel) en **0,13 ms**. Sur six essais
« dérègle puis lâche », le potentiel du mix baisse **6 fois sur 6**, y compris
en neutralisant le terme d'ancrage — c'est le mix lui-même qui s'améliore, pas
seulement le retour à l'intention.

## Structure

| Fichier | Rôle |
|---|---|
| `field.py` | **Le cœur** : potentiel, gradient analytique, découpage en bandes |
| `web/js/field.js` | Miroir JS, à garder d'accord au chiffre près (test de parité) |
| `spec.json` | **Source unique** des constantes, lue par les deux implémentations |
| `fit_target.py` | Mesure le gabarit spectral sur la bibliothèque |
| `prepare_demo.py` | Extrait une boucle multipiste depuis `decompositions/` |
| `web/js/mix.js` | Web Audio : stems → analyseurs pré-fader → bus, porte de silence |
| `web/js/motor.js` | Physique du fader motorisé + retour haptique (Vibration API) |
| `web/js/surface.js` | La X-Touch à l'écran — **même API que `web/src/lib/mcu.ts`** |
| `web/js/app.js` | La boucle fermée |
| `serve.py` | Sert la page, collecte les traces (`--record`) |

## Le passage au matériel

`surface.js` expose `setFader` · `setMeter` · `setVpotRing` · `setMuteLed` …
et émet `{type:"fader"|"mute"|"solo"|"touch"…}` : le vocabulaire MCU de
`web/src/lib/mcu.ts`, qui pilote déjà les faders motorisés en pitch-bend.
Brancher la vraie console revient à échanger l'implémentation.

Une seule notion n'a pas d'équivalent matériel dans l'autre sens : l'écart
doigt ↔ capuchon, dessiné ici parce qu'à l'écran la résistance ne peut que se
voir. Sur la X-Touch, elle se sent — c'est tout l'intérêt, et c'est ce que la
démo à l'écran ne peut qu'approcher.

## Ce qui n'est pas fait

- **Le monde appris.** Aujourd'hui le champ est analytique. La suite est un
  prédicteur conditionné par l'action, entraîné en auto-supervision sur les
  traces (`serve.py --record`) : encodeur de spectre → espace latent,
  prédicteur `(zₜ, aₜ) → ẑₜ₊₁`, tête de potentiel, et la force devient le
  gradient à travers le prédicteur. Le point qui décide de tout est
  l'anti-effondrement (VICReg ou prédicteur asymétrique + cible EMA) : sans
  lui, le modèle apprend une constante et prédit parfaitement rien.
  Le juge de réussite est déjà en place : **battre la référence analytique**,
  qui capture les phases décorrélées mais ni la compression ni le masquage
  psychoacoustique.
- Les V-Pots bougent leur anneau mais ne commandent aucun paramètre.
- Le fader master n'est pas dans le champ (il porterait le terme de niveau).
- Sur téléphone, la cible tactile des boutons fait 66×36 px : la gouttière de
  4 px limite l'extension verticale, la largeur compense. Sous les 44 px de la
  charte, assumé.
- README en anglais à écrire avant toute publication.
