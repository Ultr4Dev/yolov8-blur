{
    description = "Flake for yolo8 x86 linux amd";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    };

    outputs = {nixpkgs, ...}: let

        libPath = with pkgs; lib.makeLibraryPath [];
        system = "x86_64-linux";
        pkgs = nixpkgs.legacyPackages.${system};

    in {
        devShells.${system}.default = pkgs.mkShell {
            shellHook = "source env/bin/activate";

            packages = with pkgs; [
                python310
            ];
        };
    };
}
