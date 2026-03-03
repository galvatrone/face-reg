package UI;

import java.awt.BorderLayout;

import javax.swing.BorderFactory;
import javax.swing.JPanel;

public class AppLayout extends JPanel {
    public AppLayout(AppController controller, Runnable closeAction) {
        setLayout(new BorderLayout(20, 20));
        setBackground(BaseWindow.THEME.background);
        setBorder(BorderFactory.createEmptyBorder(22, 22, 22, 22));

        add(new HeaderPanel(controller, closeAction), BorderLayout.NORTH);
        add(new ContentPanel(controller), BorderLayout.CENTER);
    }
}
