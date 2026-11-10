# Data card

`landmarks_alphabet.csv.gz` — one row per image, produced by `scripts/extract_landmarks.py` (MediaPipe Hands 0.10.21, `static_image_mode=True`, up to 2 hands, detection confidence 0.5).
The raw images are **not** stored in the repository (32,647 images, ≈ 300 MB).

| Source | Images | Classes | Hand found | Image size |
|---|---:|---:|---:|---|
| ISL_Dataset_2 | 702 | 23 | 72.1% | 128x128 |
| Indian | 31,945 | 26 | 97.5% | 128x128 |

## Columns

| Column(s) | Meaning |
|---|---|
| `source` | `Indian` (studio, 26 letters) or `ISL_Dataset_2` (external, 23 letters) |
| `label` | the letter (folder name) |
| `file`, `frame_idx` | image file name and the number in it (consecutive numbers = neighbouring video frames) |
| `img_w`, `img_h` | image size in pixels |
| `n_hands` | hands detected: 0, 1 or 2 (rows with 0 have all-zero coordinates) |
| `h0_label`, `h0_score`, `h1_label`, `h1_score` | MediaPipe's Left/Right label and confidence for the 1st/2nd hand (kept only to reproduce the original project's features) |
| `h{0,1}_{0..20}_{x,y,z}` | 126 landmark coordinates. The two hands are stored **left-to-right by wrist position**. x, y are fractions of the image width/height, z is relative depth |

## Quality checks (all automated in `tests/test_data_integrity.py`)

* no missing values; `n_hands` ∈ {0,1,2}; labels ⊂ A–Z; coordinates inside the image range; empty hand slots are exactly zero; no duplicated (source, label, file).
* Hand found in 97.5% of studio images and 72.1% of external images; weakest studio letters: Q (61.8%), O (91.8%), C (91.9%).
* **Near-duplicates:** for 100% of sampled studio images the nearest same-letter image is ≤ 2 frames away → never use a random split on this data.
* The external dataset has no H, J, Y and only 21–46 images per letter.

## Intended use / not intended for

Training and evaluating a *static fingerspelling* recogniser and for teaching evaluation methodology. **Not** suitable for claims about real-world sign-language users, continuous signing, or other sign languages.

## Provenance and licence — TO BE COMPLETED BY THE AUTHOR

The original project files do not record where the images came from. Before publishing, add: dataset names, authors, URLs/DOIs, licences and any consent statements for `Indian` and `ISL_Dataset_2`. Until then treat the images as third-party data with unknown licence; the landmark numbers are derived data and contain no pictures or faces.
