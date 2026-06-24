import { useEffect, useRef, useState } from "react";
import { openAuthBrowser, pollDeviceAuth, startDeviceAuth } from "../api";
import { setSessionToken } from "../authSession";

interface Props {
  onAuthenticated: () => void;
}

type Phase = "opening" | "waiting" | "error";

export default function BrowserAuthGate({ onAuthenticated }: Props) {
  const [phase, setPhase] = useState<Phase>("opening");
  const [error, setError] = useState("");
  const [loginUrl, setLoginUrl] = useState("");
  const [autoOpened, setAutoOpened] = useState(true);
  const [runId, setRunId] = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    abortRef.current = controller;

    setPhase("opening");
    setError("");
    setAutoOpened(true);

    (async () => {
      try {
        const { device_code, login_url, interval } = await startDeviceAuth();
        if (controller.signal.aborted) return;
        setLoginUrl(login_url);

        // Backend opens the system browser (macOS: 'open', Windows: startfile,
        // Linux: 'xdg-open'). If that fails, the manual link below is the fallback.
        const { opened } = await openAuthBrowser(login_url);
        if (controller.signal.aborted) return;
        setAutoOpened(opened);
        setPhase("waiting");

        const token = await pollDeviceAuth(device_code, interval, controller.signal);
        setSessionToken(token);
        onAuthenticated();
      } catch (e) {
        if (controller.signal.aborted) return;
        console.error("Device flow error:", e);
        setPhase("error");
        setError(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => controller.abort();
  }, [runId, onAuthenticated]);

  const waiting = phase === "opening" || phase === "waiting";

  return (
    <div className="login-screen">
      <div className="login-shell login-shell--browser">
        <header className="login-header">
          <div className="studio-logo login-logo">
            <span className="logo-mark">✦</span>
            <span className="logo-text">llmtune</span>
            <span className="logo-badge">STUDIO</span>
          </div>
          <h1>Sign in to continue</h1>
          <p>
            A browser window has been opened for you to sign in.
            The app will continue automatically once you&apos;re done.
          </p>
        </header>

        <div className="browser-auth-body">
          {waiting ? (
            <div className="browser-auth-waiting">
              <span className="spinner" aria-hidden />
              <div>
                <strong>
                  {phase === "opening" ? "Opening browser…" : "Waiting for browser login…"}
                </strong>
                <p>
                  {phase === "opening"
                    ? "Your browser is being opened for sign-in."
                    : "Complete sign-in in your browser. This window will update automatically."}
                </p>
              </div>
            </div>
          ) : (
            <button type="button" className="btn btn-start" onClick={() => setRunId((n) => n + 1)}>
              🌐  Try Again
            </button>
          )}

          {/* Manual fallback — always available so the flow can't silently stall. */}
          {loginUrl && phase !== "error" && (
            <div className="browser-auth-manual">
              {!autoOpened && (
                <p className="form-error" style={{ marginBottom: 6 }}>
                  Couldn&apos;t open your browser automatically — use this link to sign in:
                </p>
              )}
              <a className="browser-auth-link" href={loginUrl} target="_blank" rel="noreferrer">
                Open sign-in page
              </a>
              <button
                type="button"
                className="browser-auth-copy"
                onClick={() => navigator.clipboard?.writeText(loginUrl)}
                title="Copy sign-in link"
              >
                Copy link
              </button>
            </div>
          )}

          {error && <p className="form-error">{error}</p>}

          <p className="browser-auth-hint">
            Your models and datasets never leave this machine.
          </p>
        </div>
      </div>
    </div>
  );
}
