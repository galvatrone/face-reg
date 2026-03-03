package UI;

import java.awt.Dimension;

import javax.swing.JFrame;

public class BaseWindow extends JFrame {
    protected static final Theme THEME = new Theme();

    public BaseWindow(String title) {
        super(title);
        configureWindow();
    }

    protected void configureWindow() {
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(980, 640);
        setMinimumSize(new Dimension(820, 540));
        setLocationRelativeTo(null);
    }
}
