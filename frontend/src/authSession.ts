const TOKEN_KEY = "llmtune_session_token";

// Use the same base URL detection as api.ts
function getBase(): string {
  if (typeof window !== 'undefined' && (window as any).__LLMTUNE_API_BASE__) {
    return (window as any).__LLMTUNE_API_BASE__;
  }
  if (typeof window !== 'undefined' && window.location.origin.startsWith('http')) {
    return window.location.origin;
  }
  return 'http://127.0.0.1:8765';
}

const API_BASE = getBase();

export function getSessionToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setSessionToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearSessionToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

export async function verifySession(): Promise<boolean> {
  const token = getSessionToken();
  if (!token) return false;
  try {
    const res = await fetch(`${API_BASE}/auth/verify`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    return res.ok;
  } catch {
    return false;
  }
}
