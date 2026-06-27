import Dashboard from "./components/Dashboard";
import "./styles.css";

// No login — the app opens straight into the studio.
export default function App() {
  return <Dashboard skipAuth={true} />;
}
