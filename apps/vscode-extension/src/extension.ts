import * as fs from "fs/promises";
import * as path from "path";
import { execFile, spawn, type ChildProcess } from "child_process";
import * as vscode from "vscode";

type WaggleStatus = "not-installed" | "ready" | "connected" | "error";
type McpRootKey = "servers" | "mcpServers";
type JsonObject = Record<string, unknown>;

const OUTPUT_CHANNEL = "Waggle";
const WAGGLE_SERVER_NAME = "waggle";
const DEFAULT_COMMAND = "waggle-mcp";
const DEFAULT_DB_PATH = "~/.waggle/waggle.db";
const GRAPH_STUDIO_URL = "http://127.0.0.1:8686/graph?mode=edit";

interface CommandResult {
  code: number;
  stdout: string;
  stderr: string;
}

interface ExtensionState {
  graphStudioProcess: ChildProcess | undefined;
  status: WaggleStatus;
}

export function activate(context: vscode.ExtensionContext): void {
  const output = vscode.window.createOutputChannel(OUTPUT_CHANNEL);
  const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBar.command = "waggle.showStatus";
  const state: ExtensionState = {
    graphStudioProcess: undefined,
    status: "not-installed"
  };
  context.subscriptions.push(output, statusBar, new vscode.Disposable(() => state.graphStudioProcess?.kill()));

  const append = (message: string): void => {
    output.appendLine(`[waggle] ${message}`);
  };

  const config = (): vscode.WorkspaceConfiguration => vscode.workspace.getConfiguration("waggle");
  const workspaceFolder = (): vscode.WorkspaceFolder | undefined => vscode.workspace.workspaceFolders?.[0];
  const commandPath = (): string => config().get<string>("commandPath", DEFAULT_COMMAND);

  const resolveTenantId = (): string => {
    const configured = config().get<string>("tenantId", "${workspaceFolderBasename}");
    if (configured !== "${workspaceFolderBasename}") {
      return configured;
    }
    return workspaceFolder()?.name ?? "default";
  };

  const setStatus = (status: WaggleStatus, detail = ""): void => {
    state.status = status;
    const suffix = detail ? `: ${detail}` : "";
    const labels: Record<WaggleStatus, string> = {
      "not-installed": `Waggle: Not Installed${suffix}`,
      ready: `Waggle: Ready${suffix}`,
      connected: `Waggle: Connected${suffix}`,
      error: `Waggle: Error${suffix}`
    };
    statusBar.text = labels[status];
    statusBar.show();
  };

  const buildWorkspaceServerConfig = (): JsonObject => ({
    type: "stdio",
    command: commandPath(),
    args: ["serve", "--transport", "stdio"],
    env: {
      WAGGLE_DEFAULT_TENANT_ID: resolveTenantId(),
      WAGGLE_DB_PATH: config().get<string>("dbPath", DEFAULT_DB_PATH)
    }
  });

  const execFileAsync = async (command: string, args: string[], cwd?: string): Promise<CommandResult> =>
    await new Promise((resolve, reject) => {
      execFile(command, args, { cwd, windowsHide: true }, (error, stdout, stderr) => {
        const numericCode = (error as NodeJS.ErrnoException | null)?.code;
        const code = typeof numericCode === "number" ? numericCode : 0;
        if (error && typeof (error as NodeJS.ErrnoException).code !== "number") {
          reject(error);
          return;
        }
        resolve({ code, stdout, stderr });
      });
    });

  const showOutput = (): void => output.show(true);

  const flushResult = (result: CommandResult): void => {
    if (result.stdout.trim()) {
      output.append(result.stdout);
    }
    if (result.stderr.trim()) {
      output.append(result.stderr);
    }
  };

  const updateStatusFromEnvironment = async (): Promise<boolean> => {
    try {
      const result = await execFileAsync(commandPath(), ["--version"]);
      if (result.code === 0) {
        setStatus(state.graphStudioProcess ? "connected" : "ready", result.stdout.trim());
        return true;
      }
      setStatus("error", "version check failed");
      flushResult(result);
      return false;
    } catch (error) {
      setStatus("not-installed");
      append(`CLI not available: ${String(error)}`);
      return false;
    }
  };

  const parseJsonFile = async (filePath: string): Promise<JsonObject> => {
    try {
      const raw = await fs.readFile(filePath, "utf8");
      return JSON.parse(raw) as JsonObject;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        return {};
      }
      throw error;
    }
  };

  const determineRootKey = (payload: JsonObject): McpRootKey => {
    if (isJsonObject(payload.servers)) {
      return "servers";
    }
    if (isJsonObject(payload.mcpServers)) {
      return "mcpServers";
    }
    return config().get<McpRootKey>("mcpConfigScope", "servers");
  };

  const writeWorkspaceConfig = async (): Promise<boolean> => {
    const folder = workspaceFolder();
    if (!folder) {
      void vscode.window.showWarningMessage("Open a workspace folder before enabling Waggle for this workspace.");
      return false;
    }

    const filePath = path.join(folder.uri.fsPath, ".vscode", "mcp.json");
    let existing: JsonObject;
    try {
      existing = await parseJsonFile(filePath);
    } catch (error) {
      void vscode.window.showErrorMessage(`Cannot parse existing .vscode/mcp.json: ${String(error)}`);
      setStatus("error", "invalid mcp.json");
      return false;
    }

    const rootKey = determineRootKey(existing);
    const currentRoot = isJsonObject(existing[rootKey]) ? { ...(existing[rootKey] as JsonObject) } : {};
    const waggleConfig = buildWorkspaceServerConfig();
    const previousWaggle = currentRoot[WAGGLE_SERVER_NAME];
    const actionLabel = previousWaggle ? "Update Waggle Config" : "Write Waggle Config";
    const previewPayload: JsonObject = {
      [rootKey]: {
        [WAGGLE_SERVER_NAME]: waggleConfig
      }
    };
    const detail = JSON.stringify(previewPayload, null, 2);

    append(`Prepared ${actionLabel.toLowerCase()} for ${filePath}`);
    const choice = await vscode.window.showInformationMessage(
      previousWaggle
        ? "Waggle already exists in .vscode/mcp.json. Update only the Waggle block?"
        : "Review the Waggle MCP config before writing it to .vscode/mcp.json.",
      { modal: true, detail },
      actionLabel
    );
    if (choice !== actionLabel) {
      return false;
    }

    currentRoot[WAGGLE_SERVER_NAME] = waggleConfig;
    existing[rootKey] = currentRoot;

    await fs.mkdir(path.dirname(filePath), { recursive: true });
    const serialized = `${JSON.stringify(existing, null, 2)}\n`;
    JSON.parse(serialized);
    await fs.writeFile(filePath, serialized, "utf8");

    append(`Wrote ${filePath}`);
    setStatus("ready", folder.name);
    return true;
  };

  const runDoctorInternal = async (showSuccessMessage = true): Promise<boolean> => {
    showOutput();
    append(`Running: ${commandPath()} doctor`);
    try {
      const result = await execFileAsync(commandPath(), ["doctor"], workspaceFolder()?.uri.fsPath);
      flushResult(result);
      if (result.code === 0) {
        setStatus(state.graphStudioProcess ? "connected" : "connected", "doctor ok");
        if (showSuccessMessage) {
          void vscode.window.showInformationMessage("Waggle doctor completed successfully.");
        }
        return true;
      }
      setStatus("error", "doctor warnings");
      void vscode.window.showWarningMessage("Waggle doctor reported issues. See the Waggle output channel for details.");
      return false;
    } catch (error) {
      setStatus("error", "doctor failed");
      append(`Doctor failed: ${String(error)}`);
      void vscode.window.showErrorMessage("Could not run waggle-mcp doctor.");
      return false;
    }
  };

  const installWaggle = async (showPostInstallMessage = true): Promise<boolean> => {
    const method = config().get<string>("installMethod", "pipx");
    if (method !== "pipx") {
      void vscode.window.showInformationMessage("Binary install support is reserved for a later Waggle extension update.");
      return false;
    }

    showOutput();
    append("Running: pipx install waggle-mcp");
    try {
      const result = await execFileAsync("pipx", ["install", "waggle-mcp"], workspaceFolder()?.uri.fsPath);
      flushResult(result);
      if (result.code !== 0) {
        setStatus("error", "install failed");
        void vscode.window.showErrorMessage("Waggle install failed. See the Waggle output channel for details.");
        return false;
      }
      append("Waggle installed successfully.");
      await updateStatusFromEnvironment();
      if (showPostInstallMessage) {
        void vscode.window.showInformationMessage("Waggle installed successfully.");
      }
      return true;
    } catch (error) {
      setStatus("error", "install failed");
      append(`Install failed: ${String(error)}`);
      void vscode.window.showErrorMessage("Waggle install failed. Ensure pipx is installed and available on PATH.");
      return false;
    }
  };

  const onboardWaggle = async (): Promise<void> => {
    const folder = workspaceFolder();
    if (!folder) {
      void vscode.window.showWarningMessage("Open a workspace folder before running Waggle setup.");
      return;
    }

    const proceed = await vscode.window.showInformationMessage(
      "Enable Waggle for this workspace? This will install the Waggle CLI if needed, write .vscode/mcp.json after confirmation, and run waggle-mcp doctor.",
      { modal: true },
      "Enable Waggle"
    );
    if (proceed !== "Enable Waggle") {
      return;
    }

    const available = await updateStatusFromEnvironment();
    if (!available) {
      const installed = await installWaggle(false);
      if (!installed) {
        return;
      }
    }

    const configured = await writeWorkspaceConfig();
    if (!configured) {
      return;
    }

    const doctorOk = await runDoctorInternal(false);
    if (doctorOk) {
      setStatus("connected", folder.name);
      void vscode.window.showInformationMessage("Waggle is installed, configured, and ready for this workspace.");
      return;
    }
    void vscode.window.showWarningMessage("Waggle was installed and configured, but doctor reported issues. See the Waggle output channel.");
  };

  const maybePromptInstall = async (): Promise<void> => {
    const available = await updateStatusFromEnvironment();
    if (available) {
      return;
    }
    const choice = await vscode.window.showInformationMessage(
      "Waggle is not set up in this VS Code workspace. Enable it now?",
      "Enable Waggle",
      "Open Docs"
    );
    if (choice === "Enable Waggle") {
      await onboardWaggle();
      return;
    }
    if (choice === "Open Docs") {
      await openInstallDocs();
    }
  };

  const attachProcessLogging = (child: ChildProcess, label: string): void => {
    child.stdout?.on("data", (chunk: Buffer | string) => output.append(String(chunk)));
    child.stderr?.on("data", (chunk: Buffer | string) => output.append(String(chunk)));
    child.on("exit", (code) => {
      append(`${label} exited with code ${String(code ?? 0)}.`);
      if (state.graphStudioProcess === child) {
        state.graphStudioProcess = undefined;
        void updateStatusFromEnvironment();
      }
    });
  };

  const openGraphStudio = async (): Promise<void> => {
    const available = await updateStatusFromEnvironment();
    if (!available) {
      void vscode.window.showErrorMessage("Waggle is not installed. Enable Waggle first.");
      return;
    }
    if (state.graphStudioProcess) {
      setStatus("connected", "Graph Studio");
      await vscode.env.openExternal(vscode.Uri.parse(GRAPH_STUDIO_URL));
      return;
    }

    showOutput();
    append(`Starting: ${commandPath()} graph-studio --host 127.0.0.1 --port 8686 --no-open`);
    try {
      const child = spawn(commandPath(), ["graph-studio", "--host", "127.0.0.1", "--port", "8686", "--no-open"], {
        cwd: workspaceFolder()?.uri.fsPath,
        detached: false,
        stdio: ["ignore", "pipe", "pipe"],
        windowsHide: true
      });
      state.graphStudioProcess = child;
      attachProcessLogging(child, "Graph Studio");
      setStatus("connected", "Graph Studio");
      setTimeout(() => {
        void vscode.env.openExternal(vscode.Uri.parse(GRAPH_STUDIO_URL));
      }, 1200);
    } catch (error) {
      setStatus("error", "graph studio failed");
      append(`Graph Studio failed to start: ${String(error)}`);
      void vscode.window.showErrorMessage("Could not start Waggle Graph Studio.");
    }
  };

  const exportMemory = async (): Promise<void> => {
    const folder = workspaceFolder();
    const defaultUri = folder ? vscode.Uri.file(path.join(folder.uri.fsPath, "waggle-export.abhi")) : undefined;
    const target = await vscode.window.showSaveDialog({
      defaultUri,
      filters: {
        "ABHI Export": ["abhi"]
      },
      saveLabel: "Export Waggle Memory"
    });
    if (!target) {
      return;
    }

    showOutput();
    append(`Running: ${commandPath()} export --output ${target.fsPath}`);
    try {
      const result = await execFileAsync(commandPath(), ["export", "--output", target.fsPath], folder?.uri.fsPath);
      flushResult(result);
      if (result.code !== 0) {
        setStatus("error", "export failed");
        void vscode.window.showErrorMessage("Waggle export failed. See the output channel for details.");
        return;
      }
      void vscode.window.showInformationMessage(`Waggle memory exported to ${target.fsPath}.`);
    } catch (error) {
      setStatus("error", "export failed");
      append(`Export failed: ${String(error)}`);
      void vscode.window.showErrorMessage("Could not export Waggle memory.");
    }
  };

  const openInstallDocs = async (): Promise<void> => {
    await vscode.env.openExternal(
      vscode.Uri.parse("https://github.com/Abhigyan-Shekhar/Waggle-mcp/tree/main/docs/install")
    );
  };

  const showStatus = async (): Promise<void> => {
    await updateStatusFromEnvironment();
    showOutput();
    append(`Status: ${statusBar.text}`);
  };

  context.subscriptions.push(
    vscode.commands.registerCommand("waggle.enableWorkspace", onboardWaggle),
    vscode.commands.registerCommand("waggle.install", installWaggle),
    vscode.commands.registerCommand("waggle.doctor", runDoctorInternal),
    vscode.commands.registerCommand("waggle.openGraphStudio", openGraphStudio),
    vscode.commands.registerCommand("waggle.showStatus", showStatus),
    vscode.commands.registerCommand("waggle.exportMemory", exportMemory),
    vscode.commands.registerCommand("waggle.openInstallDocs", openInstallDocs)
  );

  void maybePromptInstall();
}

export function deactivate(): void {}

function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
