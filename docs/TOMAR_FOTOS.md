# 📸 Guía para tomar fotos caseras de minerales

## Objetivo

El taller exige que el **conjunto de test** contenga fotos tomadas **con tu celular** ("otro día o con otra cámara"). Esto mide si el clasificador realmente generaliza o solo memorizó las fotos profesionales del dataset.

## Materiales

- 📱 Tu celular (cualquier modelo moderno sirve)
- 🪨 Los 4 minerales: 1 de cada uno (cuarzo, pirita, malaquita, hematita)
- 📄 Una cartulina blanca mate (o papel Kraft claro)
- 💡 Luz natural cerca de una ventana (sin sol directo)
- ⏰ 15-20 minutos

## Montaje

```
        ventana (luz indirecta)
              │
              ▼
   ┌──────────────────┐
   │                  │
   │   cartulina      │  ← sobre la mesa
   │   blanca mate    │
   │                  │
   │   🪨 cuarzo      │  ← un mineral a la vez
   │   ✨ pirita      │
   │   💚 malaquita   │
   │   ⚫ hematita    │
   │                  │
   └──────────────────┘
              ▲
              │
          📱 celular  ← perpendicular, 25-30 cm arriba
```

## Configuración del celular

- ✅ **Resolución máxima** (no uses modo "macro")
- ✅ **Enfoque automático** tocando el mineral en la pantalla
- ✅ **Cuadrícula desactivada**, **flash apagado**
- ✅ **HDR desactivado** (altera colores)
- ❌ Sin filtros ni modo retrato

## Tomas por clase (mínimo 12 cada una = 48 total)

Por cada mineral toma:

1. **6 fotos cenitales** (mirando desde arriba):
   - 2 con el mineral centrado
   - 2 desplazado a la izquierda
   - 2 desplazado a la derecha
2. **3 fotos a 30-45°** (ligeramente inclinadas):
   - Simula vista humana
3. **3 fotos con variación de luz**:
   - Cambia la posición del mineral respecto a la ventana
   - Más cerca / más lejos de la fuente de luz

**Total: 12 fotos × 4 minerales = 48 fotos**

## Tips prácticos

| ✅ Sí | ❌ No |
|---|---|
| Iluminación difusa | Flash directo |
| Fondo blanco/cartulina | Mesa con patrón o color |
| Mineral limpio | Polvo o manchas |
| Enfoque nítido | Imagen movida |
| Múltiples ángulos | Todas idénticas |
| Variar la posición | Una sola pose |
| Variar distancia luz | Siempre igual luz |

## Estructura de carpetas donde guardar

Tus fotos se deben copiar a `dataset/test/{clase}/` reemplazando/añadiendo a las existentes:

```
dataset/test/
├── cuarzo/      ← 12 fotos tuyas con nombre:  celular_001.jpg, celular_002.jpg, ...
├── pirita/      ← 12 fotos tuyas
├── malaquita/   ← 12 fotos tuyas
└── hematita/    ← 12 fotos tuyas
```

**Importante**: ejecuta `agregar_fotos_propias.py` (siguiente paso) para renombrar y validar automáticamente.

## Después de tomar las fotos

1. Pásalas al computador (cable USB, AirDrop, Google Drive, etc.)
2. Ponlas en una carpeta temporal (ej. `mis_fotos/`)
3. Ejecuta: `python3 agregar_fotos_propias.py mis_fotos/`
4. Re-ejecuta: `./run.sh` para ver el nuevo accuracy