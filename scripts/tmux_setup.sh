git clone https://github.com/tmux-plugins/tpm /root/.tmux/plugins/tpm
echo """# List of plugins
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-sensible'
set -g @continuum-restore 'on'
set -g @continuum-boot 'on'
set -g @continuum-save-interval '60'
set -g @resurrect-capture-pane-contents 'on'
 
# Initialize TMUX plugin manager (keep this line at the very bottom of tmux.conf)
run '/root/.tmux/plugins/tpm/tpm'""" >> /root/.tmux.conf

tmux source /root/.tmux.conf
