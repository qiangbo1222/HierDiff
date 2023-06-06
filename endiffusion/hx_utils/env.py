from pathlib import Path


class Env:

    def __init__(self, path='.env'):
        self.path = Path(path)
        self.lines, self.variables = self._parse_env()

    def _parse_env(self):
        lines = []
        variables = {}
        if self.path.is_file():
            with open(self.path) as f:
                for i, line in enumerate(f):
                    lines.append(line)
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, val = line.split('=')
                    variables[key] = {
                        'line': i,
                        'key': key,
                        'val': val
                    }
        return lines, variables

    def __repr__(self):
        return self.variables.__str__()

    def __getitem__(self, key):
        return self.variables[key]

    def __setitem__(self, key, val):
        if key in self.variables:
            self.variables[key]['val'] = val
        else:
            self.variables[key] = {
                'line': len(self.lines),
                'key': key,
                'val': val
            }
        self._update_line(self.variables[key])

    def __contains__(self, key):
        return key in self.variables

    def _update_line(self, var):
        line = f"{var['key']}={var['val']}\n"
        if var['line'] < len(self.lines):
            self.lines[var['line']] = line
        else:
            self.lines.append(line)

    def save(self):
        with open(self.path, 'w') as f:
            f.writelines(self.lines)