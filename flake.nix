# Adapted from
# <https://www.reddit.com/r/NixOS/comments/1p9pifu/pytorch_with_cuda_on_nixos/>
# <https://pastebin.com/PJWYht9U>
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=32f313e49e42f715491e1ea7b306a87c16fe0388";
  };
  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      pythonPackage = "python311";
      cudaSupport = true;
 
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ] (
          system:
          f (
            import nixpkgs {
              inherit system;
              config.allowUnfree = true;
              config.cudaSupport = cudaSupport;
              config.cudaVersion = "12";
            }
          )
        );
      pythonForPkgs =
        pkgs:
        pkgs.${pythonPackage}.withPackages (
          pythonPackages:
          with pythonPackages;
          [
            # pandas
            # polars
            # numpy
 
            # matplotlib
            # pytorch-bin
            # pyyaml
            # sqlalchemy
            # psycopg2-binary
            # docker
          ]
          ++ (gpuDependantPackages pkgs)
        );
 
      dependencies =
        pkgs: with pkgs; [ ];
 
      mkLibraryPath =
        pkgs:
        with pkgs;
        lib.makeLibraryPath [
          stdenv.cc.cc # numpy (on which scenedetect depends) needs C libraries
          # cudaPackages.cuda_nvrtc # libncrtc.so for cupy
          cudaPackages.cudatoolkit
          # cudaPackages.cutensor
          # cudaPackages.libcublas
          # cudaPackages.libcusolver
          # cudaPackages.cuda_cudart
        ];
 
      gpuDependantPackages =
        pkgs:
        with pkgs.${pythonPackage}.pkgs;
        if pkgs.config.cudaSupport then
          [ ]
          ++ (with pkgs; [
            cudatoolkit
            libGLU
            libGL
          ])
        else
          [ ];
    in
    {
      packages = forAllSystems (pkgs: {
        default = self.packages.${pkgs.stdenv.hostPlatform.system}.mimic-stream;
        mimic-stream = pkgs.stdenv.mkDerivation {
          name = "mimic";
          propagatedBuildInputs = [ (pythonForPkgs pkgs) ];
          dontUnpack = true;
          installPhase = "install -Dm755 ${./mimic-stream.py} $out/bin/mimic-stream";
        };
      });
      devShells = forAllSystems (pkgs: {
        default =
          let
            python = pythonForPkgs pkgs;
            cudaSupport = pkgs.config.cudaSupport;
          in
          pkgs.mkShell {
            inputsFrom = [ ];
            packages = [
              # (pkgs.writeShellScriptBin "pycharm" "tmux new -d 'pycharm-professional $1'")
              python
              # pkgs.uv
              # pkgs.ruff
              # pkgs.postgresql_18
            ]
            ++ (dependencies pkgs);
 
            shellHook = ''
              ${
                if cudaSupport then
                  ''
                    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${mkLibraryPath pkgs}:/run/opengl-driver/lib:/run/opengl-driver-32/lib"
                    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}"                                                   # For tensorflow with GPU support
                    export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
                    export EXTRA_CCFLAGS="-I/usr/include"
                  ''
                else
                  ""
              }
 
 
              export PYTHONPATH="${python}/${python.sitePackages}"
              export RUSTFLAGS='-C target-cpu=native'
              export RUST_BACKTRACE=full
 
              echo "=== PYTHON ==="
              echo
              echo "Setting PYTHONPATH to ${python}/${python.sitePackages}"
              export PYTHONPATH="${python}/${python.sitePackages}"
              echo Running $(python --version) @ $(which python) ${
                if pkgs.config.cudaSupport then "with CUDA support" else ""
              }
              echo
 
              # exec -l zsh
            '';
          };
      });
    };
}
