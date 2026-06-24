import { useEffect, useState } from "react";
import BrowserAuthGate from "./components/BrowserAuthGate";
import Dashboard from "./components/Dashboard";
import { fetchConfig } from "./api";
import { clearSessionToken, verifySession } from "./authSession";
import "./styles.css";

export default function App() {
  // Sign-in happens on Auth0's hosted page (opened in the system browser by the
  // device flow), so there is no in-app /device route anymore.
  return <StudioApp />;
}

function StudioApp() {
  const [skipAuth, setSkipAuth] = useState(false);
  const [authenticated, setAuthenticated] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const [cfg, valid] = await Promise.all([fetchConfig(), verifySession()]);
        setSkipAuth(cfg.skip_auth);
        setAuthenticated(valid);
        setReady(true);
      } catch (err) {
        console.error("[llmtune] Failed to initialize:", err);
        // Use defaults so the app can continue even if API calls fail
        setSkipAuth(false);
        setAuthenticated(false);
        setReady(true);
      }
    })();
  }, []);

  function handleLogout() {
    clearSessionToken();
    setAuthenticated(false);
  }

  if (!ready) {
    return (
      <div className="boot-screen">
        <span className="boot-logo">✦</span>
        <span>Loading llmtune…</span>
      </div>
    );
  }

  if (!skipAuth && !authenticated) {
    return <BrowserAuthGate onAuthenticated={() => setAuthenticated(true)} />;
  }

  return <Dashboard skipAuth={skipAuth} onLogout={skipAuth ? undefined : handleLogout} />;
}
