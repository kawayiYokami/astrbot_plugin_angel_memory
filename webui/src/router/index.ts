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

export const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'overview',
    component: OverviewView,
    meta: { title: '总览', icon: 'mdi-view-dashboard' },
  },
  {
    path: '/memories',
    name: 'memories',
    component: MemoryBrowseView,
    meta: { title: '记忆浏览', icon: 'mdi-brain' },
  },
  {
    path: '/profiles',
    name: 'profiles',
    component: UserProfileView,
    meta: { title: '用户画像', icon: 'mdi-account-group' },
  },
  {
    path: '/tags',
    name: 'tags',
    component: TagsDebugView,
    meta: { title: 'Tags 调试', icon: 'mdi-tag-multiple' },
  },
  {
    path: '/vector',
    name: 'vector',
    component: VectorSearchView,
    meta: { title: '向量检索', icon: 'mdi-compass' },
  },
  {
    path: '/notes',
    name: 'notes',
    component: NoteIndexView,
    meta: { title: '笔记索引', icon: 'mdi-notebook' },
  },
  {
    path: '/note-recall',
    name: 'note-recall',
    component: NoteRecallView,
    meta: { title: '笔记读取', icon: 'mdi-file-document' },
  },
  {
    path: '/import-export',
    name: 'import-export',
    component: ImportExportView,
    meta: { title: '导入导出', icon: 'mdi-swap-horizontal' },
  },
  {
    path: '/maintenance',
    name: 'maintenance',
    component: MaintenanceView,
    meta: { title: '维护状态', icon: 'mdi-wrench' },
  },
]
