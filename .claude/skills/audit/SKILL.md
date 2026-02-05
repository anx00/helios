# Spec Audit Skill

Audita codigo contra un spec/plan con verificacion de dos pasadas.

## Uso
`/audit <ruta-al-spec>`

## Procedimiento

### Pasada 1: Implementacion
1. Leer el fichero de spec completo
2. Parsear TODOS los requisitos en una checklist TodoWrite granular — nada implicito
3. Para cada requisito, buscar en el codebase si esta implementado (Grep/Glob)
4. Marcar cada item como: implementado / parcial / missing
5. Implementar todos los items missing y parciales
6. Marcar cada todo como completado al terminar

### Pasada 2: Verificacion linea-por-linea
7. Re-leer el spec DESDE CERO, linea por linea
8. Para cada requisito, hacer grep en el codigo para verificar que existe y funciona
9. Comparar el spec contra lo implementado — listar cualquier gap
10. Arreglar todos los gaps encontrados

### Output Final
11. Mostrar checklist completa con estado y referencias file:line para cada requisito
12. Formato:
    - `[x] Requisito X — src/module.py:42`
    - `[x] Requisito Y — src/engine.py:15`
    - No debe quedar ningun item sin marcar

## Reglas
- NUNCA marcar la auditoria como completa despues de una sola pasada
- Si un requisito es ambiguo, preguntar al usuario antes de asumir
- Verificar con tests si existen, o ejecutar el codigo para confirmar comportamiento
