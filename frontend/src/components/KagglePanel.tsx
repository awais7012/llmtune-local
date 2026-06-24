import { useState } from "react";
import { kaggleDownload, kaggleSearch, type KaggleDataset, type KaggleFile } from "../api";

const USER_KEY = "llmtune-kaggle-username";
const KEY_KEY = "llmtune-kaggle-key";

function fmtBytes(n: number): string {
  if (!n) return "";
  const mb = n / 1048576;
  return mb < 1024 ? `${mb.toFixed(mb < 10 ? 1 : 0)} MB` : `${(mb / 1024).toFixed(1)} GB`;
}

interface Props {
  onPick: (path: string) => void;
  onUseHf?: (id: string) => void;
}

export default function KagglePanel({ onPick, onUseHf }: Props) {
  const [username, setUsername] = useState(() => localStorage.getItem(USER_KEY) || "");
  const [key, setKey] = useState(() => localStorage.getItem(KEY_KEY) || "");
  const [showCreds, setShowCreds] = useState(() => !(localStorage.getItem(USER_KEY) && localStorage.getItem(KEY_KEY)));
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<KaggleDataset[]>([]);
  const [busy, setBusy] = useState(false);
  const [downloading, setDownloading] = useState("");
  const [files, setFiles] = useState<KaggleFile[]>([]);
  const [pickedRef, setPickedRef] = useState("");
  const [error, setError] = useState("");

  // owner/name with no spaces → almost certainly a HuggingFace dataset id.
  const looksLikeHfId = /^[\w.-]+\/[\w.-]+$/.test(query.trim());

  function saveCreds() {
    localStorage.setItem(USER_KEY, username.trim());
    localStorage.setItem(KEY_KEY, key.trim());
    setShowCreds(false);
  }

  async function search() {
    setError(""); setFiles([]); setPickedRef("");
    if (!username.trim() || !key.trim()) { setError("Enter your Kaggle username and API key first."); setShowCreds(true); return; }
    if (!query.trim()) return;
    setBusy(true);
    try {
      localStorage.setItem(USER_KEY, username.trim());
      localStorage.setItem(KEY_KEY, key.trim());
      setResults(await kaggleSearch(query.trim(), { username: username.trim(), key: key.trim() }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function download(ref: string) {
    setError(""); setDownloading(ref); setFiles([]); setPickedRef("");
    try {
      const res = await kaggleDownload(ref, { username: username.trim(), key: key.trim() });
      setPickedRef(ref);
      if (res.files.length === 0) {
        setError("Downloaded, but no CSV/JSON/JSONL/TXT files were found in this dataset.");
      } else if (res.files.length === 1) {
        onPick(res.files[0].path);
        setFiles(res.files);
      } else {
        setFiles(res.files);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDownloading("");
    }
  }

  return (
    <div>
      <button className="abtn" style={{ marginBottom: 8 }} onClick={() => setShowCreds((s) => !s)}>
        🔑 Kaggle API key {username && key ? "✓" : ""}
      </button>
      {showCreds && (
        <div className="g2" style={{ marginBottom: 8 }}>
          <div className="iw"><input placeholder="Kaggle username" value={username} onChange={(e) => setUsername(e.target.value)} /></div>
          <div className="iw">
            <input type="password" placeholder="Kaggle API key" value={key} onChange={(e) => setKey(e.target.value)} />
            <span className="iw-icon" style={{ pointerEvents: "auto", cursor: "pointer" }} onClick={saveCreds} title="Save">💾</span>
          </div>
        </div>
      )}
      {showCreds && (
        <div className="hint" style={{ marginBottom: 8 }}>
          Get a key at kaggle.com → Settings → API → Create New Token. Stored only on this machine.
        </div>
      )}

      <div className="lbl">Search Kaggle by keyword</div>
      <div className="iw" style={{ marginBottom: 6 }}>
        <input placeholder="e.g. spam, sentiment, reviews — then press Enter" value={query}
          onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") search(); }} />
        <span className="iw-icon" style={{ pointerEvents: "auto", cursor: "pointer" }} onClick={search}>{busy ? "…" : "⌕"}</span>
      </div>
      <div className="hint" style={{ marginBottom: 8 }}>Search → Download → pick a file. The dataset isn’t selected until you pick a file.</div>

      {/* If they typed an owner/name id, it's almost certainly a HuggingFace dataset. */}
      {looksLikeHfId && onUseHf && (
        <button
          className="abtn primary"
          style={{ width: "100%", justifyContent: "center", marginBottom: 8 }}
          onClick={() => onUseHf(query.trim())}
        >
          “{query.trim()}” looks like a HuggingFace dataset — use it via Local / HF →
        </button>
      )}

      {error && <div className="ds-err" style={{ marginBottom: 8 }}>✗ {error}</div>}

      {results.length > 0 && (
        <div className="mlist" style={{ maxHeight: 220 }}>
          {results.map((d) => (
            <div key={d.ref} className="mr" style={{ cursor: "default" }}>
              <div style={{ minWidth: 0 }}>
                <div className="mn" style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{d.title}</div>
                <div className="md">{d.ref} · {fmtBytes(d.size_bytes)} · ↓{d.downloads.toLocaleString()}</div>
              </div>
              <button className="abtn primary" disabled={!!downloading} onClick={() => download(d.ref)}>
                {downloading === d.ref ? "Downloading…" : "Download"}
              </button>
            </div>
          ))}
        </div>
      )}

      {files.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div className="lbl">Pick a file from {pickedRef}</div>
          <div className="mlist">
            {files.map((f) => (
              <button key={f.path} className="mr" onClick={() => onPick(f.path)}>
                <div><div className="mn">{f.name}</div><div className="md">{f.ext.toUpperCase()} · {fmtBytes(f.size_bytes)}</div></div>
                <span className="pill p-green">Use</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
