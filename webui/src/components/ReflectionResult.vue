<template>
  <div class="reflection-view">
    <div v-if="view.soul.length" class="reflect-block">
      <div class="reflect-title">当前场景心情</div>
      <div v-for="axis in view.soul" :key="axis.key" class="soul-row">
        <span class="soul-name soul-name-left">{{ axis.low }}</span>
        <v-progress-linear
          :model-value="Math.round(axis.value * 100)"
          :height="8"
          rounded
          :color="axis.color"
        />
        <span class="soul-name soul-name-right">{{ axis.high }}</span>
        <span class="soul-value">{{ axis.value.toFixed(2) }}</span>
      </div>
    </div>

    <div v-if="view.usefulIds.length" class="reflect-block">
      <div class="reflect-title">有用的记忆</div>
      <div class="reflect-plain">{{ view.usefulIds.join('、') }}</div>
    </div>

    <div v-for="group in view.groups" :key="group.title" class="reflect-block">
      <div class="reflect-title">{{ group.title }}</div>
      <div v-for="(item, index) in group.items" :key="index" class="reflect-item">
        <div class="reflect-head">
          <span v-if="item.sourceIds.length" class="reflect-source">{{
            item.sourceIds.join('、')
          }}</span>
          <span v-if="item.sourceIds.length" class="reflect-arrow">→</span>
          <v-chip size="small">{{ item.type }}</v-chip>
          <span class="reflect-judgment">{{ item.judgment }}</span>
        </div>
        <div v-if="item.reasoning" class="reflect-reasoning">{{ item.reasoning }}</div>
        <div v-if="item.tags.length" class="reflect-tags">
          <v-chip v-for="tag in item.tags" :key="tag" size="x-small" rounded="pill">
            {{ tag }}
          </v-chip>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  view: {
    soul: { key: string; low: string; high: string; value: number; color: string }[]
    usefulIds: string[]
    groups: {
      title: string
      items: {
        type: string
        judgment: string
        reasoning: string
        tags: string[]
        sourceIds: string[]
      }[]
    }[]
  }
}>()
</script>

<style scoped>
.reflection-view {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.reflect-block {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.reflect-title {
  font-size: 12px;
  font-weight: 600;
  color: rgb(var(--v-theme-secondaryText));
}

.soul-row {
  display: grid;
  grid-template-columns: 56px 1fr 56px 44px;
  align-items: center;
  gap: 10px;
}

.soul-name {
  font-size: 12px;
  color: rgb(var(--v-theme-secondaryText));
}

.soul-name-left {
  text-align: right;
}

.soul-name-right {
  text-align: left;
}

.soul-value {
  font-size: 12px;
  color: rgb(var(--v-theme-secondaryText));
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.reflect-plain {
  font-size: 13px;
  color: rgb(var(--v-theme-success));
}

.reflect-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px 10px;
  background: rgba(var(--v-theme-on-surface), 0.04);
  border-radius: 6px;
}

.reflect-head {
  display: flex;
  align-items: baseline;
  gap: 6px;
  flex-wrap: wrap;
}

.reflect-judgment {
  font-size: 13px;
  line-height: 1.6;
}

.reflect-source {
  font-size: 12px;
  opacity: 0.6;
  font-variant-numeric: tabular-nums;
}

.reflect-arrow {
  font-size: 12px;
  opacity: 0.4;
}

.reflect-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.reflect-reasoning {
  font-size: 12px;
  opacity: 0.6;
  line-height: 1.5;
  font-style: italic;
}
</style>
