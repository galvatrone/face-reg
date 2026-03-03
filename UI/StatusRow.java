package UI;

import java.awt.BorderLayout;
import java.awt.Color;

import javax.swing.BorderFactory;
import javax.swing.JLabel;
import javax.swing.SwingConstants;

public class StatusRow extends RoundedPanel {
    private final JLabel stateView;

    public StatusRow(String title, String state, Color accentColor) {
        super(BaseWindow.THEME.panel, 20);
        setLayout(new BorderLayout());
        setBorder(BorderFactory.createEmptyBorder(14, 16, 14, 16));

        JLabel titleView = new JLabel(title);
        titleView.setFont(BaseWindow.THEME.bodyFont);
        titleView.setForeground(BaseWindow.THEME.primaryText);

        stateView = new JLabel(state, SwingConstants.RIGHT);
        stateView.setFont(BaseWindow.THEME.bodyFont);
        stateView.setForeground(accentColor);

        add(titleView, BorderLayout.WEST);
        add(stateView, BorderLayout.EAST);
    }

    public void setState(String state, Color accentColor) {
        stateView.setText(state);
        stateView.setForeground(accentColor);
    }
}
