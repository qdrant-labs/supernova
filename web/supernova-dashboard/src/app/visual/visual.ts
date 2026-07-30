import { Component, ChangeDetectionStrategy } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-visual',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  changeDetection: ChangeDetectionStrategy.Eager,
  template: `
    <nav class="nav nav-pills mb-3">
      <a [routerLink]="['umap']" routerLinkActive="active" class="nav-link">UMAP</a>
      <a [routerLink]="['d3']" routerLinkActive="active" class="nav-link">D3 Visual</a>
    </nav>
    <router-outlet></router-outlet>
  `,
})
export class Visual {}
