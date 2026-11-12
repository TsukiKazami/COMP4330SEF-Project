# Annotation guideline — Hong Kong street-scene test set

Scope: the 189-image test set in `data/annotations/test_set.csv`.
Purpose: a zero-shot evaluation benchmark. Nothing here may be used to tune the
detectors; the annotation is the answer key, not training data.

---

## 1. Files and directories

| Path | Meaning |
|---|---|
| `data/images/0001.jpg …` | the 280-image candidate pool (all normalised to long side 1280) |
| `data/test_images/` | **only the 189 test images**, staged for the annotation tool |
| `data/labels_draft/` | machine drafts from YOLO11x (**not** one of the evaluated models) |
| `data/labels/` | the accepted annotation, YOLO format, one `.txt` per image |
| `data/test_images_b/`, `data/labels_b/` | the 40-image second pass, independent annotator |
| `data/annotations/attributes_form.csv` | per-image scene attributes, primary pass |
| `data/annotations/attributes_form_b.csv` | the same attributes, second pass (40 images) |

A label file is `image_id.txt` with one line per object:

```
class_index centre_x centre_y width height
```

All five numbers except `class_index` are normalised to `[0, 1]`.
`class_index` is the 0-based line number in `configs/annot_labels.txt`
(`0 = person … 79 = toothbrush`). Never reorder that file.

---

## 2. What to label

Label every instance of the 80 COCO classes that is **visible** and **identifiable**.

1. **Visibility threshold** — label an object if at least ~30 % of it is visible in
   the frame. Below that, skip it; a sliver of a bumper is not a car.
2. **Identifiability** — if you cannot tell which class it is, do not guess. Skip.
3. **Minimum size** — skip objects whose shorter side is under **8 px** at the
   original 1280-long-side resolution (roughly 4 px at the 640 inference size).
   They are below the detector's resolution and only add noise.
4. **Occlusion** — draw the box around the **estimated full extent** of the object
   when the hidden part is predictable (a car behind a railing, a pedestrian behind
   a bus). If the shape cannot be inferred, box the visible part only.
5. **Image borders** — an object cut by the frame edge is labelled if it meets the
   30 % rule; the box runs to the border.
6. **Crowds** — label each person individually. Two people overlapping are two
   boxes. A dense pavement may take several minutes; that is expected, do not
   half-label it.
7. **Box tightness** — the box should touch the object's outer extent. Do not pad
   boxes to make detection easier.

### Do not label

- classes outside the COCO 80 (traffic cones, bollards, railings, buildings,
  trees, road markings, shop signs, awnings, wheels detached from a vehicle);
- people or cars that only appear *inside* an advertisement, poster, screen or
  reflection — a photo of a person is not a person;
- shadows, reflections in puddles or windows, smoke, rain;
- objects inside another vehicle (a passenger seen through a bus window is
  visible but ambiguous; skip unless clearly identifiable and ≥30 % visible).

### Hong Kong specific mappings

| Local object | Class to use | Note |
|---|---|---|
| 的士 / private car / van | `car` | |
| 小巴 / 巴士 / coach | `bus` | minibus counts as bus |
| 貨車 / lorry | `truck` | |
| 電單車 / scooter | `motorcycle` | |
| 單車 | `bicycle` | |
| 電車 (tram) | `train` | rail vehicle; consistent for all annotators |
| 行人 / 警察 / 乘客 | `person` | |
| 交通燈 | `traffic light` | the lamp housing, not the pole |
| 停止牌 | `stop sign` | |
| 路牌、廣告牌、招牌 | **do not label** | not COCO classes |
| 輪椅、手推車、購物車 | **do not label** | not COCO classes |

If a vehicle is so deformed by occlusion or motion blur that its class is
ambiguous between `car`/`truck`, use the more specific class only when the body
type is clear; otherwise `car`.

---

## 3. Scene attributes (per image, 4 columns)

Fill these in `attributes_form.csv`. Values must be exactly one of the allowed
strings — lowercase, no spaces.

### `scale_bucket` — how large the typical object is

Look at the objects you actually annotated (ignore scenery). Take the **median
shorter-side length** of those boxes at the 1280-long-side resolution.

| Value | Rule |
|---|---|
| `small` | median shorter side < 60 px (distant traffic, pedestrians far down the road) |
| `medium` | 60–200 px |
| `large` | > 200 px (close vehicle fills a good part of the frame) |

If an image has no annotated object, use the object you would expect a detector to
be evaluated on and note it; leave the value as `medium` only as a last resort.

### `occlusion_level` — how much of the scene is hidden

| Value | Rule |
|---|---|
| `clear` | no annotated object is occluded, or only a corner is clipped |
| `partial` | 1–30 % of the typical object is hidden (fence, railing, another car) |
| `heavy` | > 30 % hidden, or at least one key object is more than half hidden (pedestrian behind a double-decker bus, vehicle behind a footbridge pillar) |

### `illumination`

| Value | Rule |
|---|---|
| `day` | direct or bright daylight, shadows visible |
| `dim` | overcast, dusk, deep shade under a flyover, strong backlight, tunnel entrance |
| `night` | street lighting or headlights are the main light source, sky dark |

### `text_density` — Traditional Chinese signage and clutter

| Value | Rule |
|---|---|
| `low` | no shop signs or at most one small sign |
| `medium` | a few signs, road markings with text, some advertising |
| `high` | dense signboards (Mong Kok, Yau Ma Tei), large banner advertising, stacked signage covering a facade |

---

## 4. Two-pass protocol for 40 images

`attributes_form_b.csv` lists 40 images, each with a **different** annotator than
the primary pass. Those 40 are also boxed twice.

Rules for the second pass:

1. Do not look at the first annotator's boxes, labels or attribute values.
2. Annotate from the image only; the YOLO11x draft is available again but it is
   the same starting point, not an answer.
3. After both passes, the two annotators compare: matching boxes (same class,
   IoU ≥ 0.5) count as agreement; every disagreement is adjudicated by the pair
   and the adjudicated value is what goes into `data/labels/`.
4. Attributes are compared label-by-label and Cohen's kappa is computed per
   attribute. Adjudication again produces the final value.

Record who annotated what: the `annotator` column is already prefilled; do not
change it.

---

## 5. Quality self-check before handing in

For every image:

- [ ] every clearly visible COCO object has a box;
- [ ] no box is for a non-COCO object or a picture inside a picture;
- [ ] no box is drastically larger or smaller than the object;
- [ ] the class is the specific one (`bus`, not `truck`, if it is a bus);
- [ ] no duplicate boxes on the same object;
- [ ] all four attribute cells are filled with an allowed value.

A quick sanity signal: the YOLO11x draft averages roughly 10–20 boxes per dense
urban frame. If your final label file has 2 boxes on a busy Mong Kok street,
you are almost certainly under-labelling.

---

## 6. Prohibitions

1. **Never** pre-annotate or check with `yolov8n/s/m` — those are the evaluated
   models; using them contaminates recall. The draft model is YOLO11x and this is
   stated in the report.
2. Do not delete or skip an image because it is hard. The test set is frozen; an
   image can only be dropped before the freeze.
3. Do not tune the annotation to what a detector would find. The answer key is
   the image, not the model output.
4. Do not edit `attributes.csv` for the 280-image pool; the test-set values live
   in the forms and are merged later.
5. Do not rename or reorder `configs/annot_labels.txt`.

---

## 7. Deliverables and naming

- boxes: `data/labels/<image_id>.txt` (189 files; an empty file is valid and means
  "no COCO object in this image");
- second pass: `data/labels_b/<image_id>.txt` (40 files);
- attributes: `data/annotations/attributes_form.csv` (189 rows) and
  `attributes_form_b.csv` (40 rows), all cells filled.

When all three are in place, the next step converts them to
`data/annotations/hk_test_coco.json` and computes the inter-annotator agreement.
