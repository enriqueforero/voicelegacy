# Publishing checklist — voicelegacy

Gate ordenado para producir un release publicable en GitHub y, opcionalmente,
en PyPI. **Cada paso ejecuta un comando real; "lo verifiqué a ojo" no cuenta.**

Auditado contra:
- `python-data-library-dev` §18.7 (pre-publish ruff)
- PEP 621 (project metadata) + PEP 639 (license-files)
- Plan A01 §6 (release engineering)

---

## 0. Pre-flight

- [ ] Working tree limpio: `git status` → `nothing to commit, working tree clean`.
- [ ] Rama actualizada: `git checkout main && git pull --rebase`.
- [ ] Branch de release creada: `git checkout -b release/vX.Y.Z`.

---

## 1. Sincronización de versión (P0 — bloquea todo)

El bug clásico: subir `pyproject.toml` a `0.2.0` y dejar `__init__.py` en
`0.1.6`. Entonces `pip show` dice una cosa y `voicelegacy.__version__`
otra. Lo descubre el primer usuario que reporte un bug "en 0.2.0" cuando
en realidad usa código de 0.1.6.

- [ ] **`pyproject.toml::version` y `voicelegacy/__init__.py::__version__` coinciden:**

      ```bash
      python -c "
      import tomllib, re
      pp = tomllib.load(open('pyproject.toml','rb'))['project']['version']
      init = re.search(r'__version__\s*=\s*[\"\\']([^\"\\']+)', open('voicelegacy/__init__.py').read()).group(1)
      assert pp == init, f'MISMATCH: pyproject={pp} init={init}'
      print(f'OK both = {pp}')
      "
      ```

      Debe imprimir `OK both = X.Y.Z`. Si falla, no avanzar.

- [ ] **Versión es la esperada** según SemVer:
  - PATCH (`X.Y.Z+1`): solo bug fixes, sin API nueva ni breaking changes.
  - MINOR (`X.Y+1.0`): nueva funcionalidad backward-compatible.
  - MAJOR (`X+1.0.0`): breaking changes. En `0.x.y` esto se relaja, pero
    se documenta explícitamente en CHANGELOG.

---

## 2. Validez de configuración

- [ ] **`pyproject.toml` parsea:**

      ```bash
      python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"
      ```

- [ ] **`.pre-commit-config.yaml` parsea:**

      ```bash
      python -c "import yaml; yaml.safe_load(open('.pre-commit-config.yaml'))"
      ```

- [ ] **Metadatos PyPI completos** (PEP 621):
  - `name`, `version`, `description`, `readme`, `requires-python`
  - `license` (SPDX expression, e.g. `"MIT"`) + `license-files`
  - `authors`, `keywords` (≥5), `classifiers` (≥10)
  - `dependencies` con upper bounds (`<X.0`) para evitar breaks transitivos
  - `[project.urls]` con al menos `Homepage`, `Repository`, `Issues`, `Changelog`

---

## 3. Lint + formato (orden estricto)

Cubre `python-data-library-dev` §18.7. Salir en el primer fallo:

```bash
# 1. Auto-fix lint
ruff check --fix --config pyproject.toml voicelegacy/ tests/ scripts/

# 2. Aplicar formato
ruff format --config pyproject.toml voicelegacy/ tests/ scripts/

# 3. Verificar lint limpio
ruff check --config pyproject.toml voicelegacy/ tests/ scripts/

# 4. Verificar formato estable
ruff format --check --config pyproject.toml voicelegacy/ tests/ scripts/
```

- [ ] Paso 3 imprime `All checks passed!`
- [ ] Paso 4 imprime `N files already formatted` sin diffs.
- [ ] Si paso 3 o 4 sigue fallando, corregir manualmente. **NO usar `# noqa`**
      sin entender la causa raíz.

---

## 4. Tests + cobertura

- [ ] **Suite completa:**

      ```bash
      pytest tests/
      ```

      Esperado: `N passed` con `N` ≥ 180.

- [ ] **Cobertura ≥ piso:**

      ```bash
      pytest tests/ --cov=voicelegacy --cov-fail-under=80
      ```

      Esperado: `Required test coverage of 80% reached. Total coverage: 85.XX%`

- [ ] **Si añadiste código**, los nuevos módulos tienen ≥ 80% individual.
      Listar con: `pytest --cov=voicelegacy --cov-report=term-missing` y
      buscar módulos < 80%.

---

## 5. Notebook

- [ ] **Regenerable desde el script:**

      ```bash
      python notebooks/build_notebook.py
      ```

      Esperado: `✅ pyproject.toml is valid TOML`, `✅ Notebook validates`,
      `Cells: 37` (o el número actual).

- [ ] **Sin divergencias hand-edited:** después de regenerar,
      `git diff notebooks/notebook_voicelegacy.ipynb` muestra SOLO diffs de
      `cell_id` UUIDs. Si hay diff en `source[]`, alguien editó el .ipynb.

- [ ] **Hook anti-bomba:**

      ```bash
      python scripts/check_no_runtime_unassign.py notebooks/notebook_voicelegacy.ipynb
      ```

      Esperado: `OK ...`

---

## 6. Pre-commit en limpio

Pre-commit DEBE pasar en una sesión sin caché. Si pasa con caché pero
falla sin caché, los hooks no protegen en CI.

```bash
pre-commit clean
pre-commit run --all-files
```

- [ ] Los 10 hooks pasan: ruff, ruff-format, trailing-whitespace,
      end-of-file-fixer, check-yaml, check-toml, check-added-large-files,
      validate-pyproject-toml, validate-notebook-schema, no-live-runtime-unassign.

- [ ] **Control negativo del anti-bomba:**

      ```bash
      python scripts/check_no_runtime_unassign.py \
        notebooks/_archive/notebook_voicelegacy_*_handedited_37cells.ipynb
      ```

      DEBE fallar con `BOMB in cell 36` y exit code 1.

---

## 7. CHANGELOG.md y PROGRESS.md

- [ ] Entrada nueva `## [X.Y.Z] — YYYY-MM-DD — Título corto` con secciones
      `Added` / `Changed` / `Fixed` / `Verified`.
- [ ] Sección `## [Unreleased]` reseteada a items que aún no se publican.
- [ ] `PROGRESS.md` actualizado si cerraste items del plan.
- [ ] Cualquier breaking change está señalado explícitamente con prefijo
      `**Breaking**:` en la descripción.

---

## 8. Build y `twine check`

```bash
# Limpiar artefactos previos
rm -rf dist/ build/ *.egg-info/

# Construir wheel + sdist
python -m build

# Verificar formato del paquete
twine check dist/*
```

- [ ] `dist/` contiene **exactamente** `voicelegacy-X.Y.Z.tar.gz` y
      `voicelegacy-X.Y.Z-py3-none-any.whl`. Cualquier `0.0.0` o versión
      vieja indica un build sucio.
- [ ] `twine check` imprime `PASSED` para ambos artefactos.
- [ ] **Inspeccionar el wheel** para confirmar que `LICENSE`, `README.md`,
      `py.typed` están incluidos:

      ```bash
      python -m zipfile -l dist/voicelegacy-*.whl | grep -E 'LICENSE|README|py.typed'
      ```

- [ ] **Tamaño del wheel razonable** (< 200 KB para un paquete de código puro
      sin assets):

      ```bash
      ls -lh dist/
      ```

---

## 9. Smoke test del paquete instalable

Antes de publicar, instalar el wheel en un venv limpio y ejecutar
el CLI básico:

```bash
python -m venv /tmp/voicelegacy-smoke
source /tmp/voicelegacy-smoke/bin/activate
pip install dist/voicelegacy-*.whl
voicelegacy --help
voicelegacy diagnose --workspace /tmp/test-ws --json || true  # workspace inexistente, debe fallar limpio
python -c "import voicelegacy; print(voicelegacy.__version__)"
deactivate
rm -rf /tmp/voicelegacy-smoke
```

- [ ] `voicelegacy --help` imprime los 6 subcomandos.
- [ ] `voicelegacy.__version__` retorna `X.Y.Z`.
- [ ] `voicelegacy diagnose` falla de forma legible (workspace no existe),
      no con `ImportError` ni stack trace de coqui-tts.

---

## 10. Release a GitHub (sin PyPI todavía)

Con los pasos 1-9 verdes:

```bash
git add -A
git commit -m "Release vX.Y.Z

- Resumen 1 línea de qué trae este release
- (Ver CHANGELOG.md)
"
git tag -a vX.Y.Z -m "vX.Y.Z — Título corto"
git push origin main --tags
```

- [ ] CI verde sobre el tag en GitHub Actions.
- [ ] **Release notes** en GitHub: copiar la sección del CHANGELOG.
- [ ] Adjuntar `dist/*.whl` y `dist/*.tar.gz` como release assets (opcional
      pero recomendable: deja un mirror estable si PyPI cae).

---

## 11. Release a PyPI (gate adicional)

**No publicar a PyPI hasta:**

- [ ] Pasos 1-10 verdes.
- [ ] **TestPyPI primero**:

      ```bash
      twine upload --repository testpypi dist/*

      # Instalar desde TestPyPI en venv aislado
      python -m venv /tmp/pypi-test
      source /tmp/pypi-test/bin/activate
      pip install --index-url https://test.pypi.org/simple/ \
                  --extra-index-url https://pypi.org/simple/ \
                  voicelegacy
      voicelegacy --help
      python -c "import voicelegacy; print(voicelegacy.__version__)"
      deactivate && rm -rf /tmp/pypi-test
      ```

- [ ] TestPyPI page muestra README renderizado correctamente
      (https://test.pypi.org/project/voicelegacy/).
- [ ] **Solo entonces:**

      ```bash
      twine upload dist/*
      ```

- [ ] Verificar https://pypi.org/project/voicelegacy/ renderiza README,
      classifiers, urls, dependencies correctamente.
- [ ] `pip install voicelegacy` en una máquina fresh funciona.

---

## 12. Post-release

- [ ] Anunciar (si aplica): README badge de versión actualizado.
- [ ] Crear branch `release/X.Y.Z+1-dev` con bump a `X.Y.(Z+1)-dev` para
      desarrollo continuo.
- [ ] Reabrir `## [Unreleased]` en CHANGELOG.

---

## Apéndice — versiones críticas

Validadas al producir `v0.2.0`:

| Herramienta | Versión |
|---|---|
| Python | 3.10 / 3.11 / 3.12 |
| ruff | ≥0.9.0 |
| pytest | ≥8.0 |
| coqui-tts | 0.27.5 |
| pre-commit | ≥3.0 |
| build | ≥1.2 |
| twine | ≥5.0 |
| nbformat | 5.x |
