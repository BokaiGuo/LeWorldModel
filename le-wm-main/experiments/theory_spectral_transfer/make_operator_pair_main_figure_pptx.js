// Package the single authoritative SVG figure onto one PowerPoint slide.
// This is a transport/layout check, not an alternative figure design.

const pptxgen = require("pptxgenjs");
const path = require("path");

const root = path.resolve(__dirname, "../..");
const figureDir = path.join(
  root,
  "results/theory_operator_alignment_iclr_oral_20260829/figures",
);

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "LeWorldModel paper artifact pipeline";
pptx.subject = "ICLR main figure: same spectrum, different transfer";
pptx.title = "Same spectrum ≠ same transfer";
pptx.company = "LeWorldModel";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Noto Sans",
  bodyFontFace: "Noto Sans",
  lang: "en-US",
};
pptx.defineSlideMaster({
  title: "FIGURE_MASTER",
  background: { color: "FFFFFF" },
  objects: [],
  slideNumber: { x: 12.8, y: 7.2, color: "FFFFFF" },
});

const slide = pptx.addSlide("FIGURE_MASTER");
slide.addImage({
  path: path.join(figureDir, "operator_pair_main_figure_figma.svg"),
  x: 0.25,
  y: 1.92,
  w: 12.833,
  h: 3.52,
});
slide.addNotes(
  "Single formal manuscript figure. Quantitative values are generated from the stored same-spectrum sweep JSON; PowerPoint preserves the authoritative SVG without adding presentation graphics.",
);

pptx.writeFile({
  fileName: path.join(figureDir, "operator_pair_main_figure.pptx"),
});
