<template>
  <v-app>
    <v-navigation-drawer v-model="drawer" :rail="rail" permanent>
      <v-list-item
        prepend-icon="mdi-brain"
        title="Angel Memory"
        subtitle="记忆管理面板"
        nav
      >
        <template v-slot:append>
          <v-btn
            icon="mdi-chevron-left"
            variant="text"
            @click="rail = !rail"
            :style="{ transform: rail ? 'rotate(180deg)' : '' }"
          />
        </template>
      </v-list-item>

      <v-divider />

      <v-list density="compact" nav>
        <v-list-item
          v-for="route in navRoutes"
          :key="route.path"
          :prepend-icon="route.meta?.icon as string"
          :title="route.meta?.title as string"
          :to="route.path"
          :active="$route.path === route.path"
          color="primary"
        />
      </v-list>
    </v-navigation-drawer>

    <v-main>
      <v-container fluid class="pa-4">
        <router-view />
      </v-container>
    </v-main>
  </v-app>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useBridge } from '@/composables/useBridge'

const router = useRouter()
const { init } = useBridge()

const drawer = ref(true)
const rail = ref(false)

const navRoutes = computed(() => router.getRoutes().filter(r => r.meta?.title))

onMounted(async () => {
  await init()
})
</script>
