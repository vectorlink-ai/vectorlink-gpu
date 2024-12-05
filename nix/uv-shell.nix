{workspace, pythonSet, mkShell, uv, stdenv, lib, parquet-tools, rust-bin, cudaPackages, linuxPackages}:
let editableOverlay = workspace.mkEditablePyprojectOverlay {
      # Use environment variable
      root = "$REPO_ROOT";
      # Optional: Only enable editable for these packages
      # members = [ "hello-world" ];
    };
    editablePythonSet = pythonSet.overrideScope editableOverlay;
    virtualenv = editablePythonSet.mkVirtualEnv "torch-experiment-dev-env" workspace.deps.all; in
mkShell {
  packages = [
    virtualenv
    uv
    parquet-tools
    (rust-bin.stable.latest.default.override {
      extensions = [ "rust-src" "rust-analyzer" "clippy" ];
    })
  ];
  shellHook = ''
    # Undo dependency propagation by nixpkgs.
    unset PYTHONPATH
    # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
    # export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${lib.makeLibraryPath [linuxPackages.nvidia_x11]}"
    export REPO_ROOT=$(git rev-parse --show-toplevel)
    export TRITON_LIBCUDA_PATH=${cudaPackages.cuda_cudart}/lib
    export LIBRARY_PATH="${lib.makeLibraryPath [cudaPackages.cuda_cudart]}/stubs"
    export LIBTORCH_USE_PYTORCH=1
  '';
}

