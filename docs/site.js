"use strict";

// The illustration shows executed depth, never synthetic benchmark results.
const loops = document.querySelector("#loops");
function updateBudget() {
  const count = Number(loops.value);
  document.querySelector("#loop-count").textContent = `${count} ${count === 1 ? "loop" : "loops"}`;
  document.querySelector("#loop-value").textContent = count;
  document.querySelector("#effective-depth").textContent = count * 4;
  document.querySelector("#budget-ratio").textContent = `${(count / 7).toFixed(2)}×`;
  document.querySelector("#regime").textContent = count < 7 ? "Under-unrolling" : count === 7 ? "Training horizon" : "Loop extrapolation";
  loops.setAttribute("aria-valuetext", `${count} loops, effective depth ${count * 4}`);
}
loops.addEventListener("input", updateBudget);
updateBudget();

const copyButton = document.querySelector("#copy-citation");
copyButton.addEventListener("click", async () => {
  const citation = document.querySelector("#bibtex");
  const status = document.querySelector("#copy-status");
  try {
    await navigator.clipboard.writeText(citation.textContent.trim());
    copyButton.textContent = "Copied!";
    status.textContent = "BibTeX copied to clipboard.";
    setTimeout(() => { copyButton.textContent = "Copy citation"; }, 2500);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = "Citation selected. Press Ctrl+C (Windows/Linux) or ⌘C (Mac) to copy.";
  }
});

// Links continue to open the original image when JavaScript is unavailable.
const dialog = document.querySelector("#figure-dialog");
const expandedFigure = document.querySelector("#expanded-figure");
let figureTrigger;
document.querySelectorAll(".figure-zoom").forEach(link => {
  link.addEventListener("click", event => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || typeof dialog.showModal !== "function") return;
    event.preventDefault();
    figureTrigger = link;
    const image = link.querySelector("img");
    expandedFigure.src = link.href;
    expandedFigure.alt = image.alt;
    document.querySelector("#figure-dialog-title").textContent = link.getAttribute("aria-label").replace(/^Enlarge /, "");
    dialog.showModal();
    document.body.classList.add("modal-open");
    const viewport = dialog.querySelector(".dialog-image-wrap");
    viewport.scrollTo(0, 0);
  });
});
document.querySelector("#close-figure").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", event => {
  const bounds = dialog.getBoundingClientRect();
  if (event.target === dialog && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) dialog.close();
});
dialog.addEventListener("close", () => {
  document.body.classList.remove("modal-open");
  if (figureTrigger) figureTrigger.focus({preventScroll: true});
});

// Group the detailed sections under the five primary navigation entries.
const navLinks = [...document.querySelectorAll(".nav-links a")];
const landmarks = navLinks.map(link => document.querySelector(link.hash));
let scrollPending = false;
function updateNavigation() {
  let active = -1;
  landmarks.forEach((section, index) => {
    if (section.getBoundingClientRect().top <= 150) active = index;
  });
  navLinks.forEach((link, index) => {
    if (index === active) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  });
  scrollPending = false;
}
window.addEventListener("scroll", () => {
  if (!scrollPending) {
    scrollPending = true;
    requestAnimationFrame(updateNavigation);
  }
}, {passive: true});
window.addEventListener("resize", updateNavigation);
updateNavigation();
