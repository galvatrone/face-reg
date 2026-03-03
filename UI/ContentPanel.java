package UI;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.DefaultComboBoxModel;
import javax.swing.ImageIcon;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import javax.swing.JTextArea;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

public class ContentPanel extends JPanel implements AppController.UiCallbacks {
    private static final int LOG_LINES = 14;
    private static final int PREVIEW_REFRESH_MS = 140;
    private static final int CONTROL_WIDTH = 390;

    private final AppController controller;
    private final Timer refreshTimer;
    private JComboBox<ProgramOption> programSelector;
    private JTextArea outputArea;
    private JTextArea logArea;
    private JLabel descriptionLabel;
    private JLabel previewLabel;
    private JLabel cameraStatusLabel;
    private StatusRow processStatus;

    public ContentPanel(AppController controller) {
        this.controller = controller;
        setOpaque(false);
        setLayout(new BorderLayout(20, 0));

        JPanel controlColumn = createControlColumn();
        controlColumn.setPreferredSize(new Dimension(CONTROL_WIDTH, 0));
        add(controlColumn, BorderLayout.WEST);
        add(createPreviewColumn(), BorderLayout.CENTER);

        refreshTimer = new Timer(PREVIEW_REFRESH_MS, new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                refreshPreview();
                refreshLogs();
            }
        });
        refreshTimer.start();

        refreshProgramDetails();
        refreshPreview();
        refreshLogs();
    }

    private JPanel createControlColumn() {
        RoundedPanel card = new RoundedPanel(BaseWindow.THEME.surface, 28);
        card.setLayout(new BorderLayout(0, 14));
        card.setBorder(BorderFactory.createEmptyBorder(18, 18, 18, 18));

        card.add(createTopControls(), BorderLayout.NORTH);
        card.add(createOutputSection(), BorderLayout.CENTER);
        card.add(createLogSection(), BorderLayout.SOUTH);
        return card;
    }

    private JPanel createTopControls() {
        JPanel panel = new JPanel();
        panel.setOpaque(false);
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

        JLabel title = new JLabel("Program");
        title.setFont(BaseWindow.THEME.sectionFont);
        title.setForeground(BaseWindow.THEME.primaryText);
        title.setAlignmentX(Component.LEFT_ALIGNMENT);

        programSelector = new JComboBox<>(controller.getProgramOptions());
        programSelector.setFont(BaseWindow.THEME.bodyFont);
        programSelector.setBackground(BaseWindow.THEME.panel);
        programSelector.setForeground(BaseWindow.THEME.primaryText);
        programSelector.setAlignmentX(Component.LEFT_ALIGNMENT);
        programSelector.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                refreshProgramDetails();
            }
        });

        descriptionLabel = new JLabel();
        descriptionLabel.setFont(BaseWindow.THEME.bodyFont);
        descriptionLabel.setForeground(BaseWindow.THEME.secondaryText);
        descriptionLabel.setAlignmentX(Component.LEFT_ALIGNMENT);

        processStatus = new StatusRow("Process", "Idle", BaseWindow.THEME.panelAccent);
        processStatus.setAlignmentX(Component.LEFT_ALIGNMENT);

        JPanel runButtons = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
        runButtons.setOpaque(false);
        runButtons.setAlignmentX(Component.LEFT_ALIGNMENT);

        ActionButton runButton = new ActionButton("Run", BaseWindow.THEME.accent, BaseWindow.THEME);
        runButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                outputArea.setText("");
                controller.startProgram(getSelectedProgram(), ContentPanel.this);
            }
        });

        ActionButton stopButton = new ActionButton("Stop", BaseWindow.THEME.softAccent, BaseWindow.THEME);
        stopButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                controller.stopProgram(ContentPanel.this);
            }
        });

        ActionButton checkButton = new ActionButton("Optimize", BaseWindow.THEME.panelAccent, BaseWindow.THEME);
        checkButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handleCheck();
            }
        });

        runButtons.add(runButton);
        runButtons.add(stopButton);
        runButtons.add(checkButton);

        JPanel manageButtons = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
        manageButtons.setOpaque(false);
        manageButtons.setAlignmentX(Component.LEFT_ALIGNMENT);

        ActionButton renameButton = new ActionButton("Rename", BaseWindow.THEME.accent, BaseWindow.THEME);
        renameButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handleRename();
            }
        });

        ActionButton deleteButton = new ActionButton("Delete", new Color(201, 77, 77), BaseWindow.THEME);
        deleteButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handleDelete();
            }
        });

        ActionButton mergeButton = new ActionButton("Merge", BaseWindow.THEME.panelAccent, BaseWindow.THEME);
        mergeButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                handleMerge();
            }
        });

        manageButtons.add(renameButton);
        manageButtons.add(deleteButton);
        manageButtons.add(mergeButton);

        panel.add(title);
        panel.add(Box.createVerticalStrut(8));
        panel.add(programSelector);
        panel.add(Box.createVerticalStrut(8));
        panel.add(descriptionLabel);
        panel.add(Box.createVerticalStrut(12));
        panel.add(processStatus);
        panel.add(Box.createVerticalStrut(12));
        panel.add(runButtons);
        panel.add(Box.createVerticalStrut(8));
        panel.add(manageButtons);
        return panel;
    }

    private JPanel createOutputSection() {
        RoundedPanel section = new RoundedPanel(BaseWindow.THEME.panel, 24);
        section.setLayout(new BorderLayout(0, 10));
        section.setBorder(BorderFactory.createEmptyBorder(14, 14, 14, 14));

        JLabel title = new JLabel("Output");
        title.setFont(BaseWindow.THEME.sectionFont);
        title.setForeground(BaseWindow.THEME.primaryText);

        outputArea = new JTextArea();
        outputArea.setEditable(false);
        outputArea.setLineWrap(true);
        outputArea.setWrapStyleWord(true);
        outputArea.setRows(9);
        outputArea.setFont(BaseWindow.THEME.monoFont);
        outputArea.setBackground(BaseWindow.THEME.surface);
        outputArea.setForeground(BaseWindow.THEME.primaryText);
        outputArea.setText("Select a program and press Run.\n");
        outputArea.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        JScrollPane scrollPane = new JScrollPane(outputArea);
        scrollPane.setBorder(BorderFactory.createEmptyBorder());

        section.add(title, BorderLayout.NORTH);
        section.add(scrollPane, BorderLayout.CENTER);
        return section;
    }

    private JPanel createLogSection() {
        RoundedPanel section = new RoundedPanel(BaseWindow.THEME.panel, 24);
        section.setLayout(new BorderLayout(0, 10));
        section.setBorder(BorderFactory.createEmptyBorder(14, 14, 14, 14));

        JLabel title = new JLabel("Recent Logs");
        title.setFont(BaseWindow.THEME.sectionFont);
        title.setForeground(BaseWindow.THEME.primaryText);

        logArea = new JTextArea();
        logArea.setEditable(false);
        logArea.setLineWrap(true);
        logArea.setWrapStyleWord(true);
        logArea.setRows(7);
        logArea.setFont(BaseWindow.THEME.monoFont);
        logArea.setBackground(BaseWindow.THEME.surface);
        logArea.setForeground(BaseWindow.THEME.primaryText);
        logArea.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));

        JScrollPane scrollPane = new JScrollPane(logArea);
        scrollPane.setBorder(BorderFactory.createEmptyBorder());

        section.add(title, BorderLayout.NORTH);
        section.add(scrollPane, BorderLayout.CENTER);
        return section;
    }

    private JPanel createPreviewColumn() {
        RoundedPanel card = new RoundedPanel(BaseWindow.THEME.surface, 28);
        card.setLayout(new BorderLayout(0, 14));
        card.setBorder(BorderFactory.createEmptyBorder(18, 18, 18, 18));

        JPanel top = new JPanel();
        top.setOpaque(false);
        top.setLayout(new BoxLayout(top, BoxLayout.Y_AXIS));

        JLabel title = new JLabel("Camera Preview");
        title.setFont(BaseWindow.THEME.sectionFont);
        title.setForeground(BaseWindow.THEME.primaryText);
        title.setAlignmentX(Component.LEFT_ALIGNMENT);

        cameraStatusLabel = new JLabel("Run 'Java UI Camera' to show the camera inside the UI.");
        cameraStatusLabel.setFont(BaseWindow.THEME.bodyFont);
        cameraStatusLabel.setForeground(BaseWindow.THEME.secondaryText);
        cameraStatusLabel.setAlignmentX(Component.LEFT_ALIGNMENT);

        top.add(title);
        top.add(Box.createVerticalStrut(6));
        top.add(cameraStatusLabel);

        RoundedPanel previewWrap = new RoundedPanel(BaseWindow.THEME.panel, 24);
        previewWrap.setLayout(new BorderLayout());
        previewWrap.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        previewLabel = new JLabel("No camera frame yet", SwingConstants.CENTER);
        previewLabel.setOpaque(true);
        previewLabel.setBackground(new Color(20, 25, 32));
        previewLabel.setForeground(BaseWindow.THEME.secondaryText);
        previewLabel.setFont(BaseWindow.THEME.bodyFont);
        previewLabel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        previewWrap.add(previewLabel, BorderLayout.CENTER);

        card.add(top, BorderLayout.NORTH);
        card.add(previewWrap, BorderLayout.CENTER);
        return card;
    }

    private ProgramOption getSelectedProgram() {
        return (ProgramOption) programSelector.getSelectedItem();
    }

    private void refreshProgramDetails() {
        ProgramOption option = getSelectedProgram();
        if (option == null) {
            return;
        }

        descriptionLabel.setText("<html>" + option.getDescription() + "</html>");
        if ("src/mainJV.py".equals(option.getScriptPath())) {
            cameraStatusLabel.setText("This mode writes frames for the Java preview.");
        } else if (option.usesCamera()) {
            cameraStatusLabel.setText("This mode uses the camera and can open an OpenCV window.");
        } else {
            cameraStatusLabel.setText("This mode works without the camera.");
        }
    }

    private void refreshLogs() {
        if (logArea == null) {
            return;
        }
        logArea.setText(controller.readLogTail(LOG_LINES));
        logArea.setCaretPosition(logArea.getDocument().getLength());
    }

    private void refreshPreview() {
        if (previewLabel == null) {
            return;
        }

        File statusFile = controller.getUiStatusFile();
        if (statusFile.exists()) {
            try {
                String status = java.nio.file.Files.readString(statusFile.toPath()).trim();
                if (!status.isEmpty()) {
                    cameraStatusLabel.setText(status);
                }
            } catch (IOException ignored) {
            }
        }

        File frameFile = controller.getUiFrameFile();
        if (!frameFile.exists()) {
            previewLabel.setIcon(null);
            previewLabel.setText("No camera frame yet");
            return;
        }

        try {
            BufferedImage image = ImageIO.read(frameFile);
            if (image == null) {
                previewLabel.setIcon(null);
                previewLabel.setText("Failed to read camera frame");
                return;
            }

            int width = Math.max(480, previewLabel.getWidth() - 8);
            int height = Math.max(320, previewLabel.getHeight() - 8);
            Image scaled = scaleToFit(image, width, height);
            previewLabel.setText("");
            previewLabel.setIcon(new ImageIcon(scaled));
        } catch (IOException exception) {
            previewLabel.setIcon(null);
            previewLabel.setText("Failed to load frame");
        }
    }

    private Image scaleToFit(BufferedImage image, int maxWidth, int maxHeight) {
        int imageWidth = Math.max(1, image.getWidth());
        int imageHeight = Math.max(1, image.getHeight());

        double widthRatio = (double) maxWidth / imageWidth;
        double heightRatio = (double) maxHeight / imageHeight;
        double scale = Math.min(widthRatio, heightRatio);
        scale = Math.max(0.01, scale);

        int targetWidth = Math.max(1, (int) Math.round(imageWidth * scale));
        int targetHeight = Math.max(1, (int) Math.round(imageHeight * scale));
        return image.getScaledInstance(targetWidth, targetHeight, Image.SCALE_SMOOTH);
    }

    private boolean ensureUtilityAllowed() {
        if (!controller.isProgramRunning()) {
            return true;
        }

        onError("Stop the running program before editing the face database.");
        return false;
    }

    private void handleRename() {
        if (!ensureUtilityAllowed()) {
            return;
        }

        FaceEntry[] entries = controller.getFaceEntries();
        if (entries.length == 0) {
            onError("No IDs found in the face database.");
            return;
        }

        JComboBox<FaceEntry> selector = new JComboBox<>(entries);
        selector.setFont(BaseWindow.THEME.bodyFont);
        Object[] content = { "Choose profile:", selector };
        int choice = JOptionPane.showConfirmDialog(
                this,
                content,
                "Rename Face",
                JOptionPane.OK_CANCEL_OPTION,
                JOptionPane.PLAIN_MESSAGE);
        if (choice != JOptionPane.OK_OPTION) {
            return;
        }

        FaceEntry selected = (FaceEntry) selector.getSelectedItem();
        if (selected == null) {
            return;
        }

        String newName = ask("Rename", "Enter new name:");
        if (isBlank(newName)) {
            return;
        }

        runUtility("src/manage_faces.py", "rename", selected.getId(), newName.trim());
    }

    private void handleDelete() {
        if (!ensureUtilityAllowed()) {
            return;
        }

        FaceEntry[] entries = controller.getFaceEntries();
        if (entries.length == 0) {
            onError("No IDs found in the face database.");
            return;
        }

        JComboBox<FaceEntry> selector = new JComboBox<>(entries);
        selector.setFont(BaseWindow.THEME.bodyFont);
        Object[] content = { "Choose profile to delete:", selector };
        int pick = JOptionPane.showConfirmDialog(
                this,
                content,
                "Delete Face",
                JOptionPane.OK_CANCEL_OPTION,
                JOptionPane.PLAIN_MESSAGE);
        if (pick != JOptionPane.OK_OPTION) {
            return;
        }

        FaceEntry selected = (FaceEntry) selector.getSelectedItem();
        if (selected == null) {
            return;
        }

        int answer = JOptionPane.showConfirmDialog(
                this,
                "Delete " + selected.toString() + "?",
                "Confirm Delete",
                JOptionPane.YES_NO_OPTION);
        if (answer != JOptionPane.YES_OPTION) {
            return;
        }

        runUtility("src/manage_faces.py", "delete", selected.getId());
    }

    private void handleMerge() {
        if (!ensureUtilityAllowed()) {
            return;
        }

        FaceEntry[] entries = controller.getFaceEntries();
        if (entries.length < 2) {
            onError("Need at least 2 IDs in the face database to merge.");
            return;
        }

        final JTextField searchField = new JTextField();
        searchField.setFont(BaseWindow.THEME.bodyFont);

        final JComboBox<FaceEntry> duplicateSelector = new JComboBox<>(entries);
        duplicateSelector.setFont(BaseWindow.THEME.bodyFont);

        final JComboBox<FaceEntry> originalSelector = new JComboBox<>(entries);
        originalSelector.setFont(BaseWindow.THEME.bodyFont);
        if (entries.length > 1) {
            originalSelector.setSelectedIndex(1);
        }

        final JLabel duplicatePreview = createFacePreviewLabel();
        final JLabel originalPreview = createFacePreviewLabel();

        JPanel mergePanel = new JPanel();
        mergePanel.setOpaque(false);
        mergePanel.setLayout(new BoxLayout(mergePanel, BoxLayout.Y_AXIS));
        mergePanel.add(new JLabel("Search by name:"));
        mergePanel.add(Box.createVerticalStrut(6));
        mergePanel.add(searchField);
        mergePanel.add(Box.createVerticalStrut(12));
        mergePanel.add(new JLabel("Duplicate profile:"));
        mergePanel.add(Box.createVerticalStrut(6));
        mergePanel.add(duplicateSelector);
        mergePanel.add(Box.createVerticalStrut(6));
        mergePanel.add(duplicatePreview);
        mergePanel.add(Box.createVerticalStrut(12));
        mergePanel.add(new JLabel("Original profile:"));
        mergePanel.add(Box.createVerticalStrut(6));
        mergePanel.add(originalSelector);
        mergePanel.add(Box.createVerticalStrut(6));
        mergePanel.add(originalPreview);

        updateFacePreview(duplicatePreview, (FaceEntry) duplicateSelector.getSelectedItem());
        updateFacePreview(originalPreview, (FaceEntry) originalSelector.getSelectedItem());

        duplicateSelector.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                updateFacePreview(duplicatePreview, (FaceEntry) duplicateSelector.getSelectedItem());
            }
        });

        originalSelector.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                updateFacePreview(originalPreview, (FaceEntry) originalSelector.getSelectedItem());
            }
        });

        searchField.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                applyFaceFilter(searchField, entries, duplicateSelector, originalSelector);
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                applyFaceFilter(searchField, entries, duplicateSelector, originalSelector);
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                applyFaceFilter(searchField, entries, duplicateSelector, originalSelector);
            }
        });

        int choice = JOptionPane.showConfirmDialog(
                this,
                mergePanel,
                "Merge Duplicate Face",
                JOptionPane.OK_CANCEL_OPTION,
                JOptionPane.PLAIN_MESSAGE);
        if (choice != JOptionPane.OK_OPTION) {
            return;
        }

        FaceEntry duplicate = (FaceEntry) duplicateSelector.getSelectedItem();
        FaceEntry original = (FaceEntry) originalSelector.getSelectedItem();
        if (duplicate == null || original == null) {
            return;
        }
        if (duplicate.getId().equals(original.getId())) {
            onError("Choose two different profiles for merge.");
            return;
        }

        int confirm = JOptionPane.showConfirmDialog(
                this,
                "Move encodings from\n" + duplicate.toString() + "\ninto\n" + original.toString() + "?",
                "Confirm Merge",
                JOptionPane.YES_NO_OPTION);
        if (confirm != JOptionPane.YES_OPTION) {
            return;
        }

        runUtility("src/manage_faces.py", "merge", duplicate.getId(), original.getId());
    }

    private JLabel createFacePreviewLabel() {
        JLabel label = new JLabel("No preview", SwingConstants.CENTER);
        label.setOpaque(true);
        label.setBackground(BaseWindow.THEME.surface);
        label.setForeground(BaseWindow.THEME.secondaryText);
        label.setPreferredSize(new Dimension(220, 150));
        label.setMinimumSize(new Dimension(220, 150));
        label.setMaximumSize(new Dimension(Integer.MAX_VALUE, 150));
        label.setBorder(BorderFactory.createEmptyBorder(6, 6, 6, 6));
        label.setAlignmentX(Component.LEFT_ALIGNMENT);
        return label;
    }

    private void applyFaceFilter(
            JTextField searchField,
            FaceEntry[] allEntries,
            JComboBox<FaceEntry> duplicateSelector,
            JComboBox<FaceEntry> originalSelector) {
        String query = searchField.getText();
        String normalized = query == null ? "" : query.trim().toLowerCase();

        DefaultComboBoxModel<FaceEntry> duplicateModel = new DefaultComboBoxModel<>();
        DefaultComboBoxModel<FaceEntry> originalModel = new DefaultComboBoxModel<>();

        for (FaceEntry entry : allEntries) {
            if (normalized.isEmpty()
                    || entry.getName().toLowerCase().contains(normalized)
                    || entry.getId().toLowerCase().contains(normalized)) {
                duplicateModel.addElement(entry);
                originalModel.addElement(entry);
            }
        }

        if (duplicateModel.getSize() == 0) {
            for (FaceEntry entry : allEntries) {
                duplicateModel.addElement(entry);
                originalModel.addElement(entry);
            }
        }

        duplicateSelector.setModel(duplicateModel);
        originalSelector.setModel(originalModel);
        if (originalModel.getSize() > 1) {
            originalSelector.setSelectedIndex(1);
        }
    }

    private void updateFacePreview(JLabel label, FaceEntry entry) {
        if (entry == null) {
            label.setIcon(null);
            label.setText("No preview");
            return;
        }

        File imageFile = controller.getPrimaryFaceImage(entry.getId());
        if (imageFile == null || !imageFile.exists()) {
            label.setIcon(null);
            label.setText("No photo for " + entry.getName());
            return;
        }

        try {
            BufferedImage image = ImageIO.read(imageFile);
            if (image == null) {
                label.setIcon(null);
                label.setText("Preview failed");
                return;
            }

            Image scaled = scaleToFit(image, 208, 138);
            label.setText("");
            label.setIcon(new ImageIcon(scaled));
        } catch (IOException exception) {
            label.setIcon(null);
            label.setText("Preview failed");
        }
    }

    private void handleCheck() {
        if (!ensureUtilityAllowed()) {
            return;
        }

        runUtility("src/check.py");
    }

    private void runUtility(String scriptPath, String... arguments) {
        String result = controller.runUtilityScript(scriptPath, arguments);
        outputArea.append("[UTILITY] " + result + "\n");
        outputArea.setCaretPosition(outputArea.getDocument().getLength());
        refreshLogs();
    }

    private String ask(String title, String message) {
        return JOptionPane.showInputDialog(this, message, title, JOptionPane.PLAIN_MESSAGE);
    }

    private boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }

    @Override
    public void onOutput(final String text) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                outputArea.append(text + "\n");
                outputArea.setCaretPosition(outputArea.getDocument().getLength());
            }
        });
    }

    @Override
    public void onStatus(final String statusText, final boolean running, final boolean cameraUsed) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                processStatus.setState(
                        statusText,
                        running ? BaseWindow.THEME.accent : BaseWindow.THEME.panelAccent);
                if (!running && !cameraUsed) {
                    refreshLogs();
                }
            }
        });
    }

    @Override
    public void onError(final String errorText) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                outputArea.append("[ERROR] " + errorText + "\n");
                outputArea.setCaretPosition(outputArea.getDocument().getLength());
                processStatus.setState("Error", Color.RED);
            }
        });
    }
}
