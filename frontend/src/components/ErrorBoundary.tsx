import { Component, type ReactNode } from "react";

/* If anything in the page throws while rendering, show what happened instead of a blank screen. */
export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: { componentStack?: string | null }) {
    console.error("VisionQuery UI error:", error, info.componentStack);
  }

  render() {
    const e = this.state.error;
    if (!e) return this.props.children;
    const translated = /removeChild|insertBefore|not a child of this node/i.test(e.message);
    return (
      <main className="crash" role="alert">
        <h1>The page stopped drawing</h1>
        {translated ? (
          <p>
            Something outside VisionQuery changed the page while it was updating, usually the browser's automatic translation or an
            extension that rewrites text. Turn translation off for this site (or try a private window), then reload.
          </p>
        ) : (
          <p>An error stopped the page from drawing. Reload to start again; your photo and questions are not saved.</p>
        )}
        <pre>{e.message}</pre>
        <button className="ask" onClick={() => location.reload()}>
          Reload
        </button>
      </main>
    );
  }
}
