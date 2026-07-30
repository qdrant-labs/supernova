import { Component, ChangeDetectionStrategy } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-user',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  changeDetection: ChangeDetectionStrategy.Eager,
  template: `
    <nav class="nav nav-pills mb-3">
      <a [routerLink]="['auth']" routerLinkActive="active" class="nav-link">Auth</a>
      <a [routerLink]="['profile']" routerLinkActive="active" class="nav-link">Profile</a>
    </nav>
    <router-outlet></router-outlet>
  `,
})
export class User {}
