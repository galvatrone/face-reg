package UI;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class AppController {
    public interface UiCallbacks {
        void onOutput(String text);

        void onStatus(String statusText, boolean running, boolean cameraUsed);

        void onError(String errorText);
    }

    private final File projectDir;
    private final File pythonExecutable;
    private final File logFile;
    private final ProgramOption[] programs;
    private Process currentProcess;
    private ProgramOption currentProgram;

    public AppController() {
        projectDir = new File(System.getProperty("user.dir"));
        pythonExecutable = resolvePythonExecutable(projectDir);
        logFile = new File(projectDir, "log.txt");
        programs = new ProgramOption[] {
            new ProgramOption("Java UI Camera", "src/mainJV.py", "Camera with recognition frame saved for the Java UI.", true),
            new ProgramOption("Main Camera", "src/main.py", "Main face recognition with live camera.", true),
            new ProgramOption("Low Camera", "src/low_main.py", "Lighter camera mode with lower processing.", true),
            new ProgramOption("Ceiling Camera", "src/ceiling_main.py", "Camera mode tuned for distant and upper angles.", true),
            new ProgramOption("Import Photos", "src/import_foto_p.py", "Imports faces from foto_p into the base.", false),
            new ProgramOption("Compare Photos", "src/compare_tools/compare_faces.py", "Compares two photos from input_photos.", false),
            new ProgramOption("Camera vs Photo", "src/compare_tools/compare_vectors.py", "Compares live camera face against one reference photo.", true)
        };
    }

    public ProgramOption[] getProgramOptions() {
        return programs;
    }

    public String getPythonCommandName() {
        return pythonExecutable.getPath();
    }

    public synchronized void startProgram(final ProgramOption option, final UiCallbacks callbacks) {
        if (option == null) {
            callbacks.onError("Program is not selected.");
            return;
        }

        stopCurrentProcess();

        File scriptFile = new File(projectDir, option.getScriptPath());
        if (!scriptFile.exists()) {
            callbacks.onError("Script not found: " + scriptFile.getPath());
            return;
        }

        ProcessBuilder builder = new ProcessBuilder(
                pythonExecutable.getPath(),
                scriptFile.getPath());
        builder.directory(projectDir);
        builder.redirectErrorStream(true);

        try {
            currentProcess = builder.start();
            currentProgram = option;
            callbacks.onStatus("Running", true, option.usesCamera());
            callbacks.onOutput("[RUN] " + option.toString());
            startOutputReader(currentProcess, callbacks);
            startExitWatcher(currentProcess, option, callbacks);
        } catch (IOException exception) {
            currentProcess = null;
            currentProgram = null;
            callbacks.onError(exception.getMessage());
        }
    }

    public synchronized void stopProgram(UiCallbacks callbacks) {
        if (currentProcess == null) {
            callbacks.onStatus("Idle", false, false);
            return;
        }

        stopCurrentProcess();
        callbacks.onOutput("[STOP] Process stopped.");
        callbacks.onStatus("Stopped", false, false);
    }

    public synchronized void stopCurrentProcess() {
        if (currentProcess != null) {
            terminateProcessTree(currentProcess);
            currentProcess = null;
            currentProgram = null;
        }
    }

    public synchronized void shutdownAndCleanup() {
        if (currentProcess != null) {
            terminateProcessTree(currentProcess);
            currentProcess = null;
            currentProgram = null;
        }

        deleteIfExists(getUiFrameFile());
        deleteIfExists(getUiStatusFile());

        File uiFramesDir = new File(projectDir, "ui_frames");
        String[] entries = uiFramesDir.list();
        if (entries != null && entries.length == 0) {
            uiFramesDir.delete();
        }
    }

    public String readLogFile() {
        if (!logFile.exists()) {
            return "log.txt not found yet.";
        }

        try {
            return Files.readString(logFile.toPath(), StandardCharsets.UTF_8);
        } catch (IOException exception) {
            return "Failed to read log.txt: " + exception.getMessage();
        }
    }

    public String readLogTail(int maxLines) {
        if (!logFile.exists()) {
            return "log.txt not found yet.";
        }

        try {
            List<String> lines = Files.readAllLines(logFile.toPath(), StandardCharsets.UTF_8);
            int fromIndex = Math.max(0, lines.size() - maxLines);
            return String.join("\n", lines.subList(fromIndex, lines.size()));
        } catch (IOException exception) {
            return "Failed to read log.txt: " + exception.getMessage();
        }
    }

    public File getUiFrameFile() {
        return new File(projectDir, "ui_frames/mainJV_frame.jpg");
    }

    public File getUiStatusFile() {
        return new File(projectDir, "ui_frames/mainJV_status.txt");
    }

    public synchronized boolean isProgramRunning() {
        return currentProcess != null;
    }

    public String runUtilityScript(String scriptRelativePath, String... arguments) {
        File scriptFile = new File(projectDir, scriptRelativePath);
        if (!scriptFile.exists()) {
            return "Script not found: " + scriptFile.getPath();
        }

        List<String> command = new ArrayList<>();
        command.add(pythonExecutable.getPath());
        command.add(scriptFile.getPath());
        for (String argument : arguments) {
            command.add(argument);
        }

        ProcessBuilder builder = new ProcessBuilder(command);
        builder.directory(projectDir);
        builder.redirectErrorStream(true);

        try {
            Process process = builder.start();
            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append('\n');
                }
            }

            int exitCode = process.waitFor();
            if (output.length() == 0) {
                output.append("Completed with exit code ").append(exitCode);
            } else {
                output.append("Exit code ").append(exitCode);
            }
            return output.toString().trim();
        } catch (IOException exception) {
            return "Failed to run script: " + exception.getMessage();
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            return "Script run interrupted.";
        }
    }

    public FaceEntry[] getFaceEntries() {
        String output = runUtilityScript("src/list_faces.py");
        if (output.startsWith("Script not found:") || output.startsWith("Failed to run script:")
                || output.startsWith("Script run interrupted.")) {
            return new FaceEntry[0];
        }

        String[] lines = output.split("\\R");
        List<FaceEntry> entries = new ArrayList<>();
        for (String line : lines) {
            String trimmed = line.trim();
            if (trimmed.isEmpty() || trimmed.startsWith("Exit code ")) {
                continue;
            }

            String[] parts = trimmed.split("\\t", -1);
            if (parts.length < 3) {
                continue;
            }
            entries.add(new FaceEntry(parts[0], parts[1], parts[2]));
        }
        return entries.toArray(new FaceEntry[0]);
    }

    public File getPrimaryFaceImage(String faceId) {
        File faceDir = new File(new File(projectDir, "faces"), faceId);
        if (!faceDir.isDirectory()) {
            return null;
        }

        File[] files = faceDir.listFiles();
        if (files == null) {
            return null;
        }

        File fallback = null;
        for (File file : files) {
            if (!file.isFile()) {
                continue;
            }
            String name = file.getName().toLowerCase();
            if (name.startsWith("main.")) {
                return file;
            }
            if (fallback == null && (name.endsWith(".jpg") || name.endsWith(".jpeg")
                    || name.endsWith(".png") || name.endsWith(".bmp") || name.endsWith(".webp"))) {
                fallback = file;
            }
        }
        return fallback;
    }

    private void startOutputReader(final Process process, final UiCallbacks callbacks) {
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        callbacks.onOutput(line);
                    }
                } catch (IOException exception) {
                    callbacks.onError(exception.getMessage());
                }
            }
        }, "python-output-reader");
        thread.setDaemon(true);
        thread.start();
    }

    private void startExitWatcher(final Process process, final ProgramOption option, final UiCallbacks callbacks) {
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    int exitCode = process.waitFor();
                    synchronized (AppController.this) {
                        if (currentProcess == process) {
                            currentProcess = null;
                            currentProgram = null;
                        }
                    }
                    callbacks.onStatus("Exit code " + exitCode, false, false);
                } catch (InterruptedException exception) {
                    Thread.currentThread().interrupt();
                    callbacks.onError("Wait interrupted for " + option.toString());
                }
            }
        }, "python-exit-watcher");
        thread.setDaemon(true);
        thread.start();
    }

    private File resolvePythonExecutable(File rootDir) {
        File venvPython = new File(rootDir, "venv/bin/python");
        if (venvPython.exists()) {
            return venvPython;
        }

        File dotVenvPython = new File(rootDir, ".venv/bin/python");
        if (dotVenvPython.exists()) {
            return dotVenvPython;
        }

        return new File("python3");
    }

    private void deleteIfExists(File file) {
        if (file.exists()) {
            file.delete();
        }
    }

    private void terminateProcessTree(Process process) {
        ProcessHandle handle = process.toHandle();

        List<ProcessHandle> descendants = handle.descendants()
                .sorted((first, second) -> Long.compare(second.pid(), first.pid()))
                .toList();

        for (ProcessHandle descendant : descendants) {
            descendant.destroy();
        }
        handle.destroy();

        waitForExit(handle, 1200);

        for (ProcessHandle descendant : descendants) {
            if (descendant.isAlive()) {
                descendant.destroyForcibly();
            }
        }
        if (handle.isAlive()) {
            handle.destroyForcibly();
        }

        waitForExit(handle, 1200);
    }

    private void waitForExit(ProcessHandle handle, long timeoutMs) {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (handle.isAlive() && System.currentTimeMillis() < deadline) {
            try {
                Thread.sleep(50);
            } catch (InterruptedException exception) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }
}
