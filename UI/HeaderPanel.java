package UI;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class HeaderPanel extends RoundedPanel {
    public HeaderPanel(AppController controller, final Runnable closeAction) {
        super(BaseWindow.THEME.surface, 28);
        setLayout(new BorderLayout());
        setBorder(BorderFactory.createEmptyBorder(18, 22, 18, 22));

        JLabel title = new JLabel("Face Recognition Control");
        title.setFont(BaseWindow.THEME.titleFont);
        title.setForeground(BaseWindow.THEME.primaryText);

        JLabel subtitle = new JLabel("Run Python tools, view camera, and check recent logs");
        subtitle.setFont(BaseWindow.THEME.bodyFont);
        subtitle.setForeground(BaseWindow.THEME.secondaryText);

        JPanel textBlock = new JPanel();
        textBlock.setOpaque(false);
        textBlock.setLayout(new BoxLayout(textBlock, BoxLayout.Y_AXIS));
        textBlock.add(title);
        textBlock.add(Box.createVerticalStrut(6));
        textBlock.add(subtitle);

        JPanel leftBlock = new JPanel(new FlowLayout(FlowLayout.LEFT, 14, 0));
        leftBlock.setOpaque(false);

        ActionButton closeButton = new ActionButton("Close", new Color(196, 74, 74), BaseWindow.THEME);
        closeButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                closeAction.run();
            }
        });

        leftBlock.add(closeButton);
        leftBlock.add(textBlock);

        JLabel info = new JLabel("Python: " + controller.getPythonCommandName() + "   |   Data: project root");
        info.setFont(BaseWindow.THEME.bodyFont);
        info.setForeground(BaseWindow.THEME.secondaryText);

        add(leftBlock, BorderLayout.WEST);
        add(info, BorderLayout.EAST);
    }
}
