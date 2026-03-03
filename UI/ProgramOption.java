package UI;

public class ProgramOption {
    private final String label;
    private final String scriptPath;
    private final String description;
    private final boolean usesCamera;

    public ProgramOption(String label, String scriptPath, String description, boolean usesCamera) {
        this.label = label;
        this.scriptPath = scriptPath;
        this.description = description;
        this.usesCamera = usesCamera;
    }

    public String getScriptPath() {
        return scriptPath;
    }

    public String getDescription() {
        return description;
    }

    public boolean usesCamera() {
        return usesCamera;
    }

    @Override
    public String toString() {
        return label;
    }
}
