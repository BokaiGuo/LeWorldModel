// Package the authoritative introduction SVG onto one PowerPoint slide.

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
pptx.subject = "Introduction figure: isospectral predictor geometry";
pptx.title = "Same predictor spectrum, different covariance-relative action";
pptx.company = "LeWorldModel";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Noto Sans",
  bodyFontFace: "Noto Sans",
  lang: "en-US",
};
pptx.defineSlideMaster({
  title: "INTRO_FIGURE_MASTER",
  background: { color: "FFFFFF" },
  objects: [],
  slideNumber: { x: 12.8, y: 7.2, color: "FFFFFF" },
});

const slide = pptx.addSlide("INTRO_FIGURE_MASTER");
slide.addImage({
  path: path.join(figureDir, "operator_pair_intro_figure_figma.svg"),
  x: 0.25,
  y: 1.70,
  w: 12.833,
  h: 3.95,
});
slide.addNotes(
  "Formal introduction schematic generated from the same analytical two-dimensional block as the paper. PowerPoint preserves the authoritative SVG without additional presentation graphics.",
);

pptx.writeFile({
  fileName: path.join(figureDir, "operator_pair_intro_figure.pptx"),
});
