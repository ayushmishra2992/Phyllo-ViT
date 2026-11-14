import Link from "next/link";

export default function LandingPage() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-black text-white">
      {/* Full-screen background image */}
      <img
        src="/demo2.png"
        alt=""
        className="absolute inset-0 h-full w-full object-cover"
      />

      {/* Dark overlay */}
     <div className="absolute inset-0 bg-black/15" />

      {/* Content */}
      <div className="relative z-10 flex min-h-screen items-center justify-center px-6">
  <div className="ml-[42%] w-full max-w-xl text-center">

    <h1 className="font-display text-5xl font-medium tracking-tight sm:text-6xl md:text-7xl">
      Phyllo-ViT
    </h1>

    <p className="font-display mx-auto mt-6 max-w-md text-base font-normal leading-relaxed text-white/90 sm:text-lg">
      Every leaf records the air it breathes. We record its
      arrangement, its texture to estimate what's in the air around it.
    </p>

    <Link
      href="/lab"
      className="mt-10 inline-flex items-center rounded-full bg-white px-8 py-3 text-sm font-semibold text-black transition-all duration-300 hover:scale-105 hover:bg-white/90"
    >
      Enter the lab
    </Link>

  </div>
</div>
    </main>
  );
}