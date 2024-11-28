{workspace, pythonSet, mkShell, uv, stdenv, lib, parquet-tools}:
let editableOverlay = workspace.mkEditablePyprojectOverlay {
      # Use environment variable
      root = "$REPO_ROOT";
      # Optional: Only enable editable for these packages
      # members = [ "hello-world" ];
    };
    editablePythonSet = pythonSet.overrideScope editableOverlay;
    virtualenv = editablePythonSet.mkVirtualEnv "hello-dev-env" workspace.deps.all; in
mkShell {
  packages = [
    virtualenv
    uv
    parquet-tools
  ];
  shellHook = ''
    # Undo dependency propagation by nixpkgs.
    unset PYTHONPATH
    # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${lib.makeLibraryPath [stdenv.cc.cc]}"
    export REPO_ROOT=$(git rev-parse --show-toplevel)
  '';
}

