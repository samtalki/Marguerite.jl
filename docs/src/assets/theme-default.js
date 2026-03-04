// Default to catppuccin-latte if no theme preference exists
if (!localStorage.getItem("documenter-theme")) {
    localStorage.setItem("documenter-theme", "catppuccin-latte");
}
