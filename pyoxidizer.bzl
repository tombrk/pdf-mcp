# PyOxidizer configuration to build a standalone executable and macOS .app bundle

def make_python_dist():
    # Use default embedded Python matching host Python (3.12 per .python-version)
    return default_python_distribution()


def make_exe():
    dist = make_python_dist()

    policy = dist.make_python_packaging_policy()
    # Include all native extension modules.
    policy.extension_module_filter = "all"
    # Prefer placing resources on the filesystem so C-extensions can load reliably.
    # This places resources relative to the executable in a lib/ dir.
    policy.resources_location = "filesystem-relative:lib"

    config = dist.make_python_interpreter_config()
    # Run our top-level module as a script so `if __name__ == "__main__"` triggers
    # and `arguably.run()` starts the server with defaults.
    config.run_module = "main"

    exe = dist.to_python_executable(
        name = "pdf-mcp",
        packaging_policy = policy,
        config = config,
    )

    # Install the current project and all of its dependencies into the embedded resources.
    # Requires running on macOS to resolve macOS wheels for native deps (e.g., PyMuPDF).
    exe.add_python_resources(exe.pip_install(["."]))

    return exe


def make_macos_app():
    exe = make_exe()

    app = starlark_tugger.MacOsApplicationBundleBuilder(bundle_name = "PDF MCP")
    app.set_info_plist_required_keys(
        display_name = "PDF MCP",
        identifier = "com.example.pdf-mcp",
        version = "0.1.0",
        signature = "????",
        executable = exe.name,
    )

    # Adopt a modern minimum by default; override via env if desired during build
    app.set_minimum_macos_version("11.0")

    # Place our built executable and its resources into the .app bundle
    app.add_macos_manifest(exe.to_file_manifest())

    # Optional: add an icon file if you have one (uncomment and provide path)
    # app.add_icon("artwork/pdf-mcp.icns")

    return app


def register_targets():
    register_target("exe", make_exe())
    register_target("mac_app", make_macos_app())


# Build the macOS .app by default when running `pyoxidizer build`
DEFAULT_TARGET = "mac_app"

