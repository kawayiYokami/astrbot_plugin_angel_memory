import type { RouteRecordRaw } from 'vue-router'
import OverviewView from '@/views/OverviewView.vue'
import MemoryBrowseView from '@/views/MemoryBrowseView.vue'
import TagsDebugView from '@/views/TagsDebugView.vue'
import VectorSearchView from '@/views/VectorSearchView.vue'
import NoteIndexView from '@/views/NoteIndexView.vue'
import NoteRecallView from '@/views/NoteRecallView.vue'
import ImportExportView from '@/views/ImportExportView.vue'
import MaintenanceView from '@/views/MaintenanceView.vue'
import UserProfileView from '@/views/UserProfileView.vue'
import SettingsView from '@/views/SettingsView.vue'
import DebugView from '@/views/DebugView.vue'

export const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'overview',
    component: OverviewView,
    meta: { title: '总览', icon: 'lucide:layout-dashboard' },
  },
  {
    path: '/memories',
    name: 'memories',
    component: MemoryBrowseView,
    meta: { title: '记忆浏览', icon: 'lucide:brain' },
  },
  {
    path: '/profiles',
    name: 'profiles',
    component: UserProfileView,
    meta: { title: '用户画像', icon: 'lucide:users' },
  },
  {
    path: '/tags',
    name: 'tags',
    component: TagsDebugView,
    meta: { title: 'Tags 调试', icon: 'lucide:tags' },
  },
  {
    path: '/vector',
    name: 'vector',
    component: VectorSearchView,
    meta: { title: '向量检索', icon: 'lucide:compass' },
  },
  {
    path: '/notes',
    name: 'notes',
    component: NoteIndexView,
    meta: { title: '笔记索引', icon: 'lucide:notebook-text' },
  },
  {
    path: '/note-recall',
    name: 'note-recall',
    component: NoteRecallView,
    meta: { title: '笔记读取', icon: 'lucide:file-text' },
  },
  {
    path: '/import-export',
    name: 'import-export',
    component: ImportExportView,
    meta: { title: '导入导出', icon: 'lucide:arrow-left-right' },
  },
  {
    path: '/maintenance',
    name: 'maintenance',
    component: MaintenanceView,
    meta: { title: '维护状态', icon: 'lucide:wrench' },
  },
  {
    path: '/debug',
    name: 'debug',
    component: DebugView,
    meta: { title: '功能调试', icon: 'lucide:flask-conical' },
  },
  {
    path: '/settings',
    name: 'settings',
    component: SettingsView,
    meta: { title: '插件设置', icon: 'lucide:settings' },
  },
]
