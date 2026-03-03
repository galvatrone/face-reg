package UI;

import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.SwingUtilities;
import javax.swing.UIManager;

public class base extends BaseWindow {
    private final AppController controller;

    public base() {
        super("Face Reg Dashboard");
        controller = new AppController();
        setContentPane(new AppLayout(controller, new Runnable() {
            @Override
            public void run() {
                closeApplication();
            }
        }));
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                closeApplication();
            }
        });
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                installLookAndFeel();
                new base().setVisible(true);
            }
        });
    }

    private static void installLookAndFeel() {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ignored) {
        }
    }

    private void closeApplication() {
        controller.shutdownAndCleanup();
        dispose();
        System.exit(0);
    }
}
