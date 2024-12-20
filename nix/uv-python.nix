{pyproject-nix, pyproject-build-systems, cudaPackages, lib, workspace, python3, callPackage, glibc, tbb_2021_11}:
let overlay = workspace.mkPyprojectOverlay {
      # Prefer prebuilt binary wheels as a package source.
      # Sdists are less likely to "just work" because of the metadata missing from uv.lock.
      # Binary wheels are more likely to, but may still require overrides for library dependencies.
      sourcePreference = "wheel"; # or sourcePreference = "sdist";
      # Optionally customise PEP 508 environment
      # environ = {
      #   platform_release = "5.10.65";
      # };
    };

    addCuda = package: package.overrideAttrs (p: {
      buildInputs = p.buildInputs ++ (with cudaPackages; [
        cudatoolkit
        cudnn
        cuda_cudart
        nccl
      ]);
      postInstall = ''
rm -f $out/lib/python3.12/site-packages/nvidia/__pycache__/__init__.cpython-312.pyc
'';
    });

    addCudaTo = packageSet: names: builtins.listToAttrs (map (name: { inherit name; value = addCuda(packageSet.${name}); }) names);

    cudaOverrides = final: prev: addCudaTo prev [
      "nvidia-cusolver-cu12"
      "nvidia-cusparse-cu12"
      "nvidia-cublas-cu12"
      "nvidia-cuda-cupti-cu12"
      "nvidia-cuda-nvrtc-cu12"
      "nvidia-cudnn-cu12"
      "nvidia-cufft-cu12"
      "nvidia-curand-cu12"
      "nvidia-nccl-cu12"
      "nvidia-nvjitlink-cu12"
      "nvidia-nvtx-cu12"
      "torch"
      "numba"
    ];
    pyprojectOverrides = final: prev: {
      numba = prev.numba.overrideAttrs (p: {
        buildInputs = p.buildInputs ++ [tbb_2021_11];
      });
    };
in
(callPackage pyproject-nix.build.packages {
  python = python3;
}).overrideScope (
  lib.composeManyExtensions [
    pyproject-build-systems.overlays.default
    overlay
    cudaOverrides
    pyprojectOverrides
  ]
)
