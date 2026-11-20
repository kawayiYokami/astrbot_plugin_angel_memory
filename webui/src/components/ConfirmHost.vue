<template>
  <v-dialog
    :model-value="state !== null"
    max-width="420"
    persistent
    @update:model-value="onToggle"
  >
    <v-card rounded="card">
      <v-card-title class="text-h6">{{ state?.title || '请确认' }}</v-card-title>
      <v-card-text class="text-body-2">{{ state?.text }}</v-card-text>
      <v-card-actions class="px-4 pb-3">
        <v-spacer />
        <v-btn variant="text" @click="settle(false)">
          {{ state?.cancelText || '取消' }}
        </v-btn>
        <v-btn
          :color="state?.danger ? 'error' : 'primary'"
          variant="flat"
          @click="settle(true)"
        >
          {{ state?.confirmText || '确定' }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup lang="ts">
import { useConfirmState } from '../composables/useConfirm'

const { state, settle } = useConfirmState()

function onToggle(visible: boolean) {
  if (!visible) settle(false)
}
</script>
