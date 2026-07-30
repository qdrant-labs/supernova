import { Routes } from '@angular/router';
import { Home } from './home/home';
import { Perf } from './perf/perf';
import { QdrantPage } from './qdrant/qdrant-page';
import { Data } from './data/data';
import { HttpData } from './data/http-data/http-data';
import { LocalData } from './data/local-data/local-data';
import { User } from './user/user';
import { UserAuth } from './user/user-auth/user-auth';
import { UserProfile } from './user/user-profile/user-profile';
import { Visual } from './visual/visual';
import { Umap } from './visual/umap/umap';
import { D3Visual } from './visual/d3-visual/d3-visual';
import { CursorDev } from './cursor/cursor-dev';
import { SupernovaTab } from './supernova/supernova-tab';

export const routes: Routes = [
  { path: '', redirectTo: 'home', pathMatch: 'full' },
  { path: 'home', component: Home },
  { path: 'perf', component: Perf },
  { path: 'qdrant', component: QdrantPage },
  {
    path: 'data',
    component: Data,
    children: [
      { path: '', redirectTo: 'http', pathMatch: 'full' },
      { path: 'http', component: HttpData },
      { path: 'local', component: LocalData },
    ],
  },
  {
    path: 'user',
    component: User,
    children: [
      { path: '', redirectTo: 'auth', pathMatch: 'full' },
      { path: 'auth', component: UserAuth },
      { path: 'profile', component: UserProfile },
    ],
  },
  {
    path: 'visual',
    component: Visual,
    children: [
      { path: '', redirectTo: 'd3', pathMatch: 'full' },
      { path: 'umap', component: Umap },
      { path: 'd3', component: D3Visual },
    ],
  },
  { path: 'dev/cursor', component: CursorDev },
  { path: 'supernova', component: SupernovaTab },
];
