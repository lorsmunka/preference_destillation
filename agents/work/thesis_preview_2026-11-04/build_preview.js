// thesis.md -> a .docx preview in the NIK format (guide 2.3): A4, margins 40/25/25 mm + 35 mm binding side,
// Times New Roman 12 pt, 1.5 line spacing, justified; chapter titles 14 pt capitals, centred, on a new page;
// sub-headings 12 pt bold, left; page numbers top centre 20 mm from the edge, counted from the table of contents.
// The chapter 6 talking-point block is left out (raw material, not thesis text).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle,
  AlignmentType, HeadingLevel, PageNumber, Header, TableOfContents, ShadingType, LevelFormat, PageBreak,
} = require("docx");

const SOURCE = process.argv[2];
const OUTPUT = process.argv[3];
const SOURCE_DIR = path.dirname(SOURCE);
const mm = (value) => Math.round(value * 56.6929);
const TEXT_WIDTH = 11906 - mm(35) - mm(25);
const MAX_IMAGE_PIXELS = 540;
const FONT = "Times New Roman";
const GREY = "7F7F7F";

const lines = fs.readFileSync(SOURCE, "utf8").replace(/\r\n/g, "\n").split("\n");
// A chapter title: "N. Title" (any case; the heading style prints it in capitals, guide 2.3), starting with a capital
// and not ending in a colon, so a numbered step such as "2. lépés:" is not taken for a chapter.
function isChapter(line) {
  const match = line.match(/^\d{1,2}\. (.+)$/);
  if (!match) return false;
  const title = match[1].split("(")[0].trim();
  return title.length >= 5 && /^[A-ZÁÉÍÓÖŐÚÜŰ]/.test(title) && !title.endsWith(":");
}
const isSubsection = (line) => /^\d+\.\d+(\.\d+)? \S/.test(line);

function inlineRuns(text, options = {}) {
  return text.split(/(`[^`]+`)/).filter(Boolean).map((part) => part.startsWith("`")
    ? new TextRun({ text: part.slice(1, -1), font: "Consolas", size: options.size ? options.size - 2 : 22 })
    : new TextRun({ text: part, ...options }));
}

const body = (text) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 }, children: inlineRuns(text) });
const note = (text) => new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text, italics: true, color: GREY })] });

function pngSize(file) {
  const buffer = fs.readFileSync(file);
  return { buffer, width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
}

function image(relativePath) {
  const file = path.resolve(SOURCE_DIR, relativePath);
  if (!fs.existsSync(file)) return note(`[missing image ${relativePath}]`);
  const { buffer, width, height } = pngSize(file);
  const scale = Math.min(1, MAX_IMAGE_PIXELS / width);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: buffer, transformation: { width: Math.round(width * scale), height: Math.round(height * scale) } })] });
}

function table(rows) {
  const cells = rows.filter((row, index) => index !== 1).map((row) => row.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim()));
  const columns = cells[0].length;
  const lengths = Array.from({ length: columns }, (_, column) => Math.max(4, ...cells.map((row) => (row[column] || "").length)));
  const total = lengths.reduce((sum, value) => sum + value, 0);
  const widths = lengths.map((value) => Math.floor(TEXT_WIDTH * value / total));
  widths[widths.length - 1] += TEXT_WIDTH - widths.reduce((sum, value) => sum + value, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
  return new Table({
    width: { size: TEXT_WIDTH, type: WidthType.DXA }, columnWidths: widths,
    rows: cells.map((row, rowIndex) => new TableRow({ tableHeader: rowIndex === 0, children: row.map((cell, column) => new TableCell({
      width: { size: widths[column], type: WidthType.DXA },
      borders: { top: border, bottom: border, left: border, right: border },
      shading: rowIndex === 0 ? { type: ShadingType.CLEAR, fill: "E7E6E6", color: "auto" } : undefined,
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({ spacing: { line: 240, after: 0 }, children: inlineRuns(cell, { size: 20, bold: rowIndex === 0 }) })],
    })) })),
  });
}

function code(block) {
  return block.map((line) => new Paragraph({ spacing: { line: 240, after: 0 }, shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" },
    children: [new TextRun({ text: line || " ", font: "Consolas", size: 18 })] }));
}

function convert(sourceLines) {
  const blocks = [];
  let index = 0;
  let inTalkingPoints = false;
  while (index < sourceLines.length) {
    const raw = sourceLines[index];
    const line = raw.trim();
    if (inTalkingPoints && !isChapter(line)) { index += 1; continue; }
    inTalkingPoints = false;
    if (line.startsWith("[TALKING POINTS")) {
      blocks.push(note("[A fejezet vázlatpontjai (Claude, 2026-11-04) az előnézetből kimaradtak.]"));
      inTalkingPoints = true; index += 1; continue;
    }
    if (line === "") { index += 1; continue; }
    const fence = line.match(/^(`{3,})/);
    if (fence) {
      const content = [];
      index += 1;
      while (index < sourceLines.length && !new RegExp(`^\`{${fence[1].length},}\\s*$`).test(sourceLines[index].trim())) {
        content.push(sourceLines[index]); index += 1;
      }
      blocks.push(...code(content), new Paragraph({ spacing: { after: 120 }, children: [] }));
      index += 1; continue;
    }
    if (line.startsWith("|")) {
      const rows = [];
      while (index < sourceLines.length && sourceLines[index].trim().startsWith("|")) { rows.push(sourceLines[index]); index += 1; }
      blocks.push(table(rows), new Paragraph({ spacing: { after: 120 }, children: [] }));
      continue;
    }
    const picture = line.match(/^!\[[^\]]*\]\(([^)]+)\)$/);
    if (picture) blocks.push(image(picture[1]));
    else if (isChapter(line)) blocks.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(line)] }));
    else if (/^\d+\.\d+\.\d+ \S/.test(line)) blocks.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(line)] }));
    else if (isSubsection(line)) blocks.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(line)] }));
    else if (/^\d+\.\d+\. ábra\./.test(line)) blocks.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: line, size: 22 })] }));
    else if (/^\d+\.\d+\. táblázat\./.test(line)) blocks.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 }, children: [new TextRun({ text: line, size: 22 })] }));
    else if (line.startsWith("- ")) blocks.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 60 }, children: inlineRuns(line.slice(2)) }));
    else if (/^\[\d+\] /.test(line)) blocks.push(new Paragraph({ alignment: AlignmentType.LEFT, indent: { left: mm(10), hanging: mm(10) }, spacing: { after: 60 }, children: [new TextRun(line)] }));
    else if (line.startsWith("[")) blocks.push(note(line));
    else blocks.push(body(line));
    index += 1;
  }
  return blocks;
}

const firstChapter = lines.findIndex((line) => line.trim() === "1. BEVEZETÉS");
const front = lines.slice(0, firstChapter);
const main = lines.slice(firstChapter);
const titleLine = front.find((line) => line.startsWith("Magyar cím:")) || "";
const title = titleLine.replace("Magyar cím:", "").trim();

const centred = (text, size, options = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text, size, ...options })] });
const placeholderPage = (name) => [centred(name.toUpperCase(), 28, { bold: true }), note(`[Helyőrző, ${name.toLowerCase()} (1 oldal), a kari sablon szerint.]`), new Paragraph({ children: [new PageBreak()] })];

const cover = [
  new Paragraph({ spacing: { before: mm(30) * 2 }, children: [] }),
  centred("NEUMANN JÁNOS INFORMATIKAI KAR", 32, { bold: true }),
  new Paragraph({ spacing: { before: 1200 }, children: [] }),
  centred("SZAKDOLGOZAT", 40, { bold: true }),
  new Paragraph({ spacing: { before: 600 }, children: [] }),
  centred(title, 28),
  new Paragraph({ spacing: { before: 3000 }, children: [] }),
  new Paragraph({ children: [new TextRun({ text: "OE-NIK 2026", size: 24 }), new TextRun({ text: "\tHallgató neve: Laczkó Örs", size: 24 })], tabStops: [{ type: "left", position: mm(40) }] }),
  new Paragraph({ children: [new TextRun({ text: "\tHallgató törzskönyvi száma: [kitöltendő]", size: 24 })], tabStops: [{ type: "left", position: mm(40) }] }),
  new Paragraph({ children: [new PageBreak()] }),
];

const abstractBlocks = [centred("ABSZTRAKT", 28, { bold: true }), ...convert(front.slice(1))];

const pageSetup = {
  page: { size: { width: 11906, height: 16838 }, margin: { top: mm(40), bottom: mm(25), left: mm(35), right: mm(25), header: mm(20), footer: mm(12) } },
};

const document = new Document({
  creator: "Laczkó Örs",
  title,
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 24 }, paragraph: { spacing: { line: 360 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, allCaps: true }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 360 }, keepNext: true, pageBreakBefore: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: mm(8), hanging: mm(5) } } } }] }] },
  sections: [
    { properties: pageSetup, children: [...cover, ...placeholderPage("Feladatlap"), ...placeholderPage("Hallgatói nyilatkozat"), ...placeholderPage("Konzultációs napló"), ...abstractBlocks] },
    {
      properties: { ...pageSetup, page: { ...pageSetup.page, pageNumbers: { start: 1 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT] })] })] }) },
      children: [centred("TARTALOMJEGYZÉK", 28, { bold: true }), new TableOfContents("Tartalomjegyzék", { hyperlink: true, headingStyleRange: "1-3" }), ...convert(main)],
    },
  ],
});

Packer.toBuffer(document).then((buffer) => { fs.writeFileSync(OUTPUT, buffer); console.log("written", OUTPUT, buffer.length, "bytes"); });
