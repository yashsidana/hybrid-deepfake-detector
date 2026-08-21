// Slow-drifting blurred gradient blobs + a faint grid, purely CSS-driven
// (see styles/index.css's .bg-fx rules) so it costs nothing on the main
// thread. Mounted once in App.jsx behind every page.
export default function AnimatedBackground() {
  return (
    <div className="bg-fx" aria-hidden="true">
      <div className="grid" />
      <div className="blob blob-1" />
      <div className="blob blob-2" />
      <div className="blob blob-3" />
    </div>
  );
}
