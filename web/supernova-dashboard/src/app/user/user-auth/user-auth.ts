import { Component, ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'app-user-auth',
  changeDetection: ChangeDetectionStrategy.Eager,
  template: `
    <div class="card">
      <div class="card-body">
        <h3 class="h5">User Auth</h3>
        <p class="mb-0">Authentication tab preserved from the source dashboard.</p>
      </div>
    </div>
  `,
})
export class UserAuth {}
