import { Component, ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'app-home',
  changeDetection: ChangeDetectionStrategy.Eager,
  template: `
    <div class="card">
      <div class="card-body">
        <h2 class="h4">Home</h2>
        <p class="mb-0">Welcome to the Supernova dashboard.</p>
      </div>
    </div>
  `,
})
export class Home {}
