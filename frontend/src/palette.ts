/* Boxes are coloured by the question's targets, in a fixed order validated for
   colour-vision deficiency (light and dark sets live in styles.css as --s1..--s4).
   Class names always appear in labels, tooltips and the legend, so colour is
   never the only cue. A fifth target and beyond fall back to ink. */
export const SLOTS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];
export const OTHER = "var(--ink-2)";

export function slotFor(index: number): string {
  return index < SLOTS.length ? SLOTS[index] : OTHER;
}

export const FRIENDLY: Record<string, string> = {
  pedestrian: "pedestrian",
  people: "person sitting or riding",
  bicycle: "bicycle",
  car: "car",
  van: "van",
  truck: "truck",
  tricycle: "tricycle",
  "awning-tricycle": "awning tricycle",
  bus: "bus",
  motor: "motorbike",
};

export const FRIENDLY_PLURAL: Record<string, string> = {
  pedestrian: "pedestrians",
  people: "people sitting or riding",
  bicycle: "bicycles",
  car: "cars",
  van: "vans",
  truck: "trucks",
  tricycle: "tricycles",
  "awning-tricycle": "awning tricycles",
  bus: "buses",
  motor: "motorbikes",
};
