package UI;

public class FaceEntry {
    private final String id;
    private final String name;
    private final String encodingCount;

    public FaceEntry(String id, String name, String encodingCount) {
        this.id = id;
        this.name = name;
        this.encodingCount = encodingCount;
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    @Override
    public String toString() {
        return name + " [" + id + "] (" + encodingCount + ")";
    }
}
