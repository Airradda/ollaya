{
  description = "Development environment for Ollaya";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
      ...
    }:
    let
      forEachSystem = nixpkgs.lib.genAttrs [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      pkgsFor =
        system:
        import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
          config.allowUnfree = true;
        };
      ortDistributions = {
        aarch64-darwin = {
          target = "aarch64-apple-darwin+coreml";
          hash = "sha256-aTSHTi6VNXbZwdtH/xrznGLE9CINvm+YjhMfcoeWdMc=";
        };
        aarch64-linux = {
          target = "aarch64-unknown-linux-gnu";
          hash = "sha256-BqBQq5E3zLMkIdDLSenM9y2eGKsK64+NA40bXMhEs1o=";
        };
        x86_64-linux = {
          target = "x86_64-unknown-linux-gnu";
          hash = "sha256-5FT3EPikn1OqW0/1HjRUrhg1d35DHGw1xSVc5vIF/Wg=";
        };
      };
      # Keep these pins in sync with scripts/llama-cpp.sh.
      llamaRuntimes = {
        aarch64-darwin = {
          kind = "darwin-arm64";
          asset = "llama-b11146-bin-macos-arm64.tar.gz";
          hash = "sha256-GtP57/gO252+9CWa1WTRcgYS737qSPpK/tDlT189VxE=";
        };
        aarch64-linux = {
          kind = "linux-arm64";
          asset = "llama-b11146-bin-ubuntu-arm64.tar.gz";
          hash = "sha256-Su2m/miDFUfkm3+odgc4PKU1Kz1yyl9w1S7SZfWMEx8=";
        };
        x86_64-linux = {
          kind = "linux-amd64";
          asset = "llama-b11146-bin-ubuntu-cuda-13.4-x64.tar.gz";
          hash = "sha256-FgPZwApLbqyCmMXHhozbCAo6wxlIqx5FdEHXHOJ03X4=";
        };
      };
      version = (builtins.fromTOML (builtins.readFile ./Cargo.toml)).workspace.package.version;
      mkOllaya =
        system:
        let
          pkgs = pkgsFor system;
          rust = pkgs.rust-bin.stable.latest.minimal;
          rustPlatform = pkgs.makeRustPlatform {
            cargo = rust;
            rustc = rust;
          };
          ortDistribution = ortDistributions.${system};
          ortArchive = pkgs.fetchurl {
            url = "https://cdn.pyke.io/0/pyke:ort-rs/ms@1.28.0/${ortDistribution.target}.tar.lzma2";
            inherit (ortDistribution) hash;
          };
          ort =
            pkgs.runCommand "onnxruntime-${ortDistribution.target}" { nativeBuildInputs = [ pkgs.xz ]; }
              ''
                mkdir -p "$out"
                xz --format=raw --lzma2=dict=64MiB -dc ${ortArchive} | tar -xf - -C "$out"
              '';
          source = pkgs.lib.cleanSourceWith {
            src = ./.;
            filter =
              path: type:
              let
                name = baseNameOf path;
              in
              name != "flake.nix" && name != "flake.lock" && pkgs.lib.cleanSourceFilter path type;
          };
          llamaRuntimeSpec = llamaRuntimes.${system};
          llamaArchive = pkgs.fetchurl {
            url = "https://github.com/ggml-org/llama.cpp/releases/download/b11146/${llamaRuntimeSpec.asset}";
            inherit (llamaRuntimeSpec) hash;
          };
          cudaLibraries = pkgs.lib.optionals (system == "x86_64-linux") [
            pkgs.cudaPackages_13.cuda_cudart
            pkgs.cudaPackages_13.libcublas
          ];
          llamaRuntime = pkgs.stdenvNoCC.mkDerivation {
            pname = "ollaya-llama-runtime";
            inherit version;
            dontUnpack = true;

            nativeBuildInputs = pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
              pkgs.autoPatchelfHook
            ];
            buildInputs =
              pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [ pkgs.stdenv.cc.cc.lib ] ++ cudaLibraries;
            autoPatchelfIgnoreMissingDeps = pkgs.lib.optionals (system == "x86_64-linux") [
              "libcuda.so.1"
            ];

            installPhase = ''
              runHook preInstall

              llamaCache="$TMPDIR/ollaya-package/llama.cpp-b11146"
              mkdir -p "$llamaCache"
              cp ${llamaArchive} "$llamaCache/${llamaRuntimeSpec.asset}"
              OLLAYA_CACHE="$TMPDIR/ollaya-package" \
                ${source}/scripts/llama-cpp.sh "${llamaRuntimeSpec.kind}" "$out/lib/ollaya/llama" \
                  "$out/share/doc/ollaya/llama.cpp-THIRD_PARTY_NOTICES"
            ''
            + pkgs.lib.optionalString (system == "x86_64-linux") ''
              OLLAYA_CACHE="$TMPDIR/ollaya-package" \
                ${source}/scripts/llama-cpp.sh linux-amd64-cuda "$TMPDIR/ollaya-llama-cuda"
              cp "$TMPDIR/ollaya-llama-cuda/libggml-cuda.so" "$out/lib/ollaya/llama/"
            ''
            + ''
              runHook postInstall
            '';
          };
          ollayaBinary = rustPlatform.buildRustPackage {
            pname = "ollaya-unwrapped";
            inherit version;
            src = source;

            cargoLock.lockFile = ./Cargo.lock;
            cargoBuildFlags = [
              "-p"
              "ollaya"
            ];
            cargoTestFlags = [ "--workspace" ];

            OLLAYA_BUILD_VERSION = version;
            ORT_LIB_LOCATION = ort;
            SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";

            nativeBuildInputs = pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
              pkgs.autoPatchelfHook
            ];
            buildInputs = pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
              pkgs.stdenv.cc.cc.lib
            ];
          };
        in
        pkgs.runCommand "ollaya-${version}"
          {
            nativeBuildInputs = [ pkgs.makeWrapper ];
            meta = {
              description = "Run open decision models locally";
              homepage = "https://ollaya.dev";
              license = pkgs.lib.licenses.asl20;
              mainProgram = "ollaya";
            };
          }
          ''
            mkdir -p "$out/bin"
            makeWrapper ${ollayaBinary}/bin/ollaya "$out/bin/ollaya" \
              --set-default OLLAYA_LIBRARY_PATH "${llamaRuntime}/lib/ollaya" \
              --set-default SSL_CERT_FILE "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt" \
              ${pkgs.lib.optionalString (
                system == "x86_64-linux"
              ) ''--prefix LD_LIBRARY_PATH : "/run/opengl-driver/lib"''}
          '';
    in
    {
      apps = forEachSystem (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/ollaya";
        };
      });

      packages = forEachSystem (system: {
        default = mkOllaya system;
        ollaya = mkOllaya system;
      });

      devShells = forEachSystem (
        system:
        let
          pkgs = pkgsFor system;
          rust = pkgs.rust-bin.stable.latest.default.override {
            extensions = [
              "rust-analyzer"
              "rust-src"
            ];
          };
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.cmake
              pkgs.curl
              pkgs.file
              rust
              pkgs.nodejs_24
              pkgs.pkg-config
              pkgs.shellcheck
              pkgs.unzip
              pkgs.uv
              pkgs.zstd
            ]
            ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
              pkgs.glib
              pkgs.gtk3
              pkgs.libayatana-appindicator
              pkgs.librsvg
              pkgs.openssl
              pkgs.webkitgtk_4_1
              pkgs.xdotool
            ];
          };
        }
      );

      formatter = forEachSystem (system: nixpkgs.legacyPackages.${system}.nixfmt);
    };
}
