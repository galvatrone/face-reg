package UI;

import java.awt.Component;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JLabel;

public class InfoCard extends RoundedPanel {
    public InfoCard(String label, String value, String description) {
        super(BaseWindow.THEME.surface, 26);
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        JLabel labelView = new JLabel(label);
        labelView.setFont(BaseWindow.THEME.bodyFont);
        labelView.setForeground(BaseWindow.THEME.secondaryText);
        labelView.setAlignmentX(Component.LEFT_ALIGNMENT);

        JLabel valueView = new JLabel(value);
        valueView.setFont(BaseWindow.THEME.metricFont);
        valueView.setForeground(BaseWindow.THEME.primaryText);
        valueView.setAlignmentX(Component.LEFT_ALIGNMENT);

        JLabel descriptionView = new JLabel("<html>" + description + "</html>");
        descriptionView.setFont(BaseWindow.THEME.bodyFont);
        descriptionView.setForeground(BaseWindow.THEME.secondaryText);
        descriptionView.setAlignmentX(Component.LEFT_ALIGNMENT);

        add(labelView);
        add(Box.createVerticalStrut(12));
        add(valueView);
        add(Box.createVerticalStrut(8));
        add(descriptionView);
    }
}
