import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";
import "@fontsource/spectral/latin-300.css";
import "@fontsource/spectral/latin-300-italic.css";
import "@fontsource/spectral/latin-400.css";
import "@fontsource/spectral/latin-400-italic.css";
import "@fontsource/spectral/latin-600.css";
import "@fontsource/kalam/latin-300.css";
import "@fontsource/kalam/latin-400.css";
import "./styles.css";

// apply the saved theme before first paint
try {
  const t = JSON.parse(localStorage.getItem("sq.theme") ?? '"auto"');
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
} catch {
  /* storage unavailable */
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
);
