#set text(font: "poppins")
#show link: it => box(
  fill: blue.lighten(80%),
  inset: 3pt,
  radius: 3pt,
  stroke: 0.5pt + blue,
  it
)
#show link: set text(
  font: "JetBrains Mono",
)

#show raw: set text(
  font: "JetBrains Mono",
  size: 1em,
)

= Data Specs
== MoNuSAC
- ignore class 0
- 40x magnification -> 0.25 mpp
- tile to 256x256px with 15% overlap
- categories:
  0. Ambiguous
  1. Epithelial
  2. Lymphocyte
  3. Macrophage
  4. Neutrophil
- tissues:
  0. Breast
  1. Kidney
  2. Lung
  3. Prostate


)
#grid(
  columns: 500pt,
  align: center + horizon,
  grid(
    columns: (auto, auto),
    gutter: 10pt,
    image("./assets/monusac_tis_dist.png"),
    image("./assets/monusac_cat_dist.png"),
  ),
  image("./assets/monusac_tist_cat_map.png", width: 250pt),
)

== #link("https://dakomura.github.io/SegPath/")[SegPath]
- HE image file: ``` {antigen}_{celltype}_{slideID}_{posx}_{posy}_HE.png```
- Mask image file: ``` {antigen}_{celltype}_{slideID}_{posx}_{posy}_mask.png```
- 984x984 px.
- posX and posY are the leftmost position in WSI coordinate.
- Mask files store binary segmentation mask (background: 0, target: 1). 
  - literally, it's stored as 1 and not 255
  - ``` visible_mask = mask * 255```

= Preprocessing