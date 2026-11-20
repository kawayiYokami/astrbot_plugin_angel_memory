<template>
  <v-snackbar
    :key="current?.id ?? 0"
    :model-value="current !== null"
    :color="current ? kindStyle(current.kind).color : undefined"
    :timeout="2800"
    location="bottom right"
    @update:model-value="onToggle"
  >
    <div class="d-flex align-center ga-3">
      <v-icon :icon="current ? kindStyle(current.kind).icon : 'lucide:info'" />
      <span>{{ current?.text }}</span>
    </div>
  </v-snackbar>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useToastQueue, type ToastKind } from '../composables/useToast'

const { items, shift } = useToastQueue()

const current = computed(() => items.value[0] ?? null)

function kindStyle(kind: ToastKind) {
  switch (kind) {
    case 'success':
      return { color: 'success', icon: 'lucide:circle-check' }
    case 'error':
      return { color: 'error', icon: 'lucide:circle-alert' }
    case 'warning':
      return { color: 'warning', icon: 'lucide:triangle-alert' }
    default:
      return { color: 'info', icon: 'lucide:info' }
  }
}

function onToggle(visible: boolean) {
  if (!visible) shift()
}
</script>
